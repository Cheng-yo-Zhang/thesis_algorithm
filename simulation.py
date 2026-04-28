from typing import List, Dict

from config import Config
from models import Node, VehicleState, Route, Solution
from problem import ChargingSchedulingProblem


def initialize_fleet(cfg: Config, depot: Node) -> List[VehicleState]:
    """初始化車隊 — 所有車輛在 t=0 位於 depot 待命"""
    fleet: List[VehicleState] = []
    vid = 0

    for i in range(cfg.NUM_MCS_SLOW):
        fleet.append(VehicleState(
            vehicle_id=vid, vehicle_type='mcs_slow',
            position=depot, available_time=0.0,
            remaining_energy=cfg.MCS_CAPACITY,
        ))
        vid += 1

    for i in range(cfg.NUM_MCS_FAST):
        fleet.append(VehicleState(
            vehicle_id=vid, vehicle_type='mcs_fast',
            position=depot, available_time=0.0,
            remaining_energy=cfg.MCS_CAPACITY,
        ))
        vid += 1

    for i in range(cfg.NUM_UAV):
        fleet.append(VehicleState(
            vehicle_id=vid, vehicle_type='uav',
            position=depot, available_time=0.0,
            remaining_energy=cfg.UAV_DELIVERABLE_ENERGY,
        ))
        vid += 1

    return fleet


def simulate_slot_execution(
    solution: Solution,
    vehicle_states: List[VehicleState],
    problem: ChargingSchedulingProblem,
    slot_end_time: float
) -> None:
    """
    模擬 slot 內車輛執行 — 更新 vehicle states

    對每台車, 按照 solution 中指派的 route suffix,
    模擬此 slot 期間已完成/正在執行的任務, 更新 position/time/energy
    """
    c = problem.cfg

    # 建立 vehicle_id -> route 映射
    route_map: Dict[int, Route] = {}
    for r in solution.mcs_routes:
        route_map[r.vehicle_id] = r
    for r in solution.uav_routes:
        route_map[r.vehicle_id] = r

    for vs in vehicle_states:
        route = route_map.get(vs.vehicle_id)
        if route is None or len(route.nodes) == 0:
            # 閒置車輛: 不移動
            vs.available_time = max(vs.available_time, slot_end_time)
            continue

        # 模擬逐節點執行
        current_time = vs.available_time
        current_energy = vs.remaining_energy
        current_pos = vs.position

        committed: List[Node] = []

        for i, node in enumerate(route.nodes):
            vehicle = vs.vehicle_type
            travel_time = problem.calculate_travel_time(current_pos, node, vehicle)
            dist_type = 'manhattan' if problem._is_mcs(vehicle) else 'euclidean'
            dist = problem.calculate_distance(current_pos, node, dist_type)

            arrival = current_time + travel_time
            setup, connect, disconnect = problem._get_service_overhead(vehicle)
            service_start = max(arrival + setup + connect, node.ready_time)
            _, charging_time = problem.get_charging_profile(vehicle, node.node_type, node.demand, node=node)
            departure = service_start + charging_time + disconnect

            if departure <= slot_end_time:
                # 完成服務
                node.status = 'served'
                if problem._is_mcs(vehicle):
                    current_energy -= (dist * c.MCS_ENERGY_CONSUMPTION + node.demand)
                else:
                    # UAV: 更新 node SOC 與剩餘需求
                    actual_delivery = problem.compute_uav_delivery(node)
                    node.ev_current_soc = node.ev_current_soc + actual_delivery / c.EV_BATTERY_CAPACITY
                    node.residual_demand = node.original_demand - actual_delivery
                    node.uav_served = True
                    current_energy -= actual_delivery
                current_time = departure
                current_pos = node
            else:
                # 尚未完成: 加入 committed (frozen-prefix 延伸)
                if arrival < slot_end_time:
                    # 已出發或已到達但未完成 => committed
                    node.status = 'in_service'
                    committed.append(node)
                    if problem._is_mcs(vehicle):
                        current_energy -= (dist * c.MCS_ENERGY_CONSUMPTION + node.demand)
                    else:
                        # UAV committed: 更新 node 狀態與能量
                        actual_delivery = problem.compute_uav_delivery(node)
                        node.ev_current_soc = node.ev_current_soc + actual_delivery / c.EV_BATTERY_CAPACITY
                        node.residual_demand = node.original_demand - actual_delivery
                        node.uav_served = True
                        current_energy -= actual_delivery
                    current_time = departure
                    current_pos = node
                else:
                    # 尚未出發 => 保持 assigned, 可被下一輪重規劃
                    break

        # 更新車輛狀態
        vs.position = current_pos
        vs.available_time = current_time
        vs.remaining_energy = current_energy
        vs.committed_nodes = committed


def simulate_dispatch_window(
    solution: Solution,
    vehicle_states: List[VehicleState],
    problem: ChargingSchedulingProblem,
    execution_end_time: float
) -> None:
    """
    Advance each active vehicle through the current dispatch window and
    update its release state for the next re-optimization.
    """
    c = problem.cfg

    route_map: Dict[int, Route] = {}
    for r in solution.mcs_routes:
        route_map[r.vehicle_id] = r
    for r in solution.uav_routes:
        route_map[r.vehicle_id] = r

    for vs in vehicle_states:
        route = route_map.get(vs.vehicle_id)
        if route is None or len(route.nodes) == 0:
            vs.available_time = max(vs.available_time, execution_end_time)
            continue

        current_time = vs.available_time
        current_energy = vs.remaining_energy
        current_pos = vs.position
        # Preserve existing committed nodes from previous slots (still in-service)
        new_committed: List[Node] = list(vs.committed_nodes)
        new_committed_departures: List[float] = list(vs.committed_departures)

        for node in route.nodes:
            vehicle = vs.vehicle_type
            travel_time = problem.calculate_travel_time(current_pos, node, vehicle)
            dist_type = 'manhattan' if problem._is_mcs(vehicle) else 'euclidean'
            dist = problem.calculate_distance(current_pos, node, dist_type)

            arrival = current_time + travel_time
            setup, connect, disconnect = problem._get_service_overhead(vehicle)
            service_start = max(arrival + setup + connect, node.ready_time)
            _, charging_time = problem.get_charging_profile(vehicle, node.node_type, node.demand, node=node)
            departure = service_start + charging_time + disconnect

            if departure <= execution_end_time:
                node.status = 'served'
                node.served_by_vehicle_id = vs.vehicle_id
                node.served_by_vehicle_type = vs.vehicle_type
                if problem._is_mcs(vehicle):
                    if not c.MCS_UNLIMITED_ENERGY:
                        current_energy -= (dist * c.MCS_ENERGY_CONSUMPTION + node.demand)
                else:
                    actual_delivery = problem.compute_uav_delivery(node)
                    node.ev_current_soc = node.ev_current_soc + actual_delivery / c.EV_BATTERY_CAPACITY
                    node.residual_demand = node.original_demand - actual_delivery
                    node.uav_served = True
                    current_energy -= actual_delivery
                current_time = departure
                current_pos = node
            elif arrival < execution_end_time:
                node.status = 'in_service'
                node.served_by_vehicle_id = vs.vehicle_id
                node.served_by_vehicle_type = vs.vehicle_type
                new_committed.append(node)
                new_committed_departures.append(departure)
                if problem._is_mcs(vehicle):
                    if not c.MCS_UNLIMITED_ENERGY:
                        current_energy -= (dist * c.MCS_ENERGY_CONSUMPTION + node.demand)
                else:
                    actual_delivery = problem.compute_uav_delivery(node)
                    node.ev_current_soc = node.ev_current_soc + actual_delivery / c.EV_BATTERY_CAPACITY
                    node.residual_demand = node.original_demand - actual_delivery
                    node.uav_served = True
                    current_energy -= actual_delivery
                current_time = departure
                current_pos = node
            else:
                # arrival >= execution_end_time
                if current_time < execution_end_time:
                    # 已出發但尚未抵達 → en-route, 加入 committed
                    node.status = 'in_service'
                    node.served_by_vehicle_id = vs.vehicle_id
                    node.served_by_vehicle_type = vs.vehicle_type
                    new_committed.append(node)
                    new_committed_departures.append(departure)
                    if problem._is_mcs(vehicle):
                        if not c.MCS_UNLIMITED_ENERGY:
                            current_energy -= (dist * c.MCS_ENERGY_CONSUMPTION + node.demand)
                    else:
                        actual_delivery = problem.compute_uav_delivery(node)
                        node.ev_current_soc = node.ev_current_soc + actual_delivery / c.EV_BATTERY_CAPACITY
                        node.residual_demand = node.original_demand - actual_delivery
                        node.uav_served = True
                        current_energy -= actual_delivery
                    current_time = departure
                    current_pos = node
                break

        vs.position = current_pos
        vs.available_time = current_time
        vs.remaining_energy = current_energy
        vs.committed_nodes = new_committed
        vs.committed_departures = new_committed_departures
