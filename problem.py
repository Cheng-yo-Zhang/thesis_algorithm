import numpy as np
import random
from typing import List, Tuple, Optional

from config import Config
from models import Node, VehicleState, Route, Solution, DistanceMatrix


class ChargingSchedulingProblem:
    """充電排程問題 — 批次調度 (Batch Dispatching)"""

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        c = self.cfg

        self.nodes: List[Node] = []
        self.depot: Node = None
        self.dist_matrix: Optional[DistanceMatrix] = None
        self._next_node_id: int = 1

        random.seed(c.RANDOM_SEED)
        np.random.seed(c.RANDOM_SEED)

        self.depot = Node(
            id=0, x=c.DEPOT_X, y=c.DEPOT_Y,
            demand=0, ready_time=0, due_date=1e9,
            service_time=0, node_type='depot', status='served'
        )

    # ==================== 輔助方法 ====================
    def _is_mcs(self, vehicle: str) -> bool:
        return vehicle in ('mcs_fast', 'mcs_slow')

    def _get_service_overhead(self, vehicle: str) -> Tuple[float, float, float]:
        c = self.cfg
        if vehicle == 'uav':
            return (0.0, 0.0, 0.0)
        else:
            return (c.MCS_ARRIVAL_SETUP_MIN, c.MCS_CONNECT_MIN, c.MCS_DISCONNECT_MIN)

    def _get_vehicle_speed(self, vehicle: str) -> float:
        c = self.cfg
        return c.UAV_SPEED if vehicle == 'uav' else c.MCS_SPEED

    def _get_vehicle_capacity(self, vehicle: str) -> float:
        c = self.cfg
        return c.UAV_DELIVERABLE_ENERGY if vehicle == 'uav' else c.MCS_CAPACITY

    def _get_charging_power(self, vehicle: str) -> float:
        c = self.cfg
        if vehicle == 'uav':
            return c.UAV_CHARGE_POWER
        elif vehicle == 'mcs_fast':
            return c.MCS_FAST_POWER
        else:
            return c.MCS_SLOW_POWER

    def _energy_to_depot(self, node: Node) -> float:
        """計算從 node 返回 depot 所需行駛能耗 (kWh)"""
        dist = abs(node.x - self.depot.x) + abs(node.y - self.depot.y)  # manhattan
        return dist * self.cfg.MCS_ENERGY_CONSUMPTION

    def _sample_truncated_demand(self, low: float, high: float) -> float:
        """以 rejection sampling 產生截斷常態需求值。"""
        c = self.cfg
        for _ in range(1000):
            demand = np.random.normal(c.DEMAND_MEAN, c.DEMAND_STD)
            if low <= demand <= high:
                return demand
        return float(np.clip(c.DEMAND_MEAN, low, high))

    # ==================== 需求生成 (Static Single-Batch) ====================
    def generate_requests(self) -> List[Node]:
        """生成 cfg.N_REQUESTS 個靜態請求 (批次於 t=0 ~ DELTA_T 期間到達)。"""
        c = self.cfg

        new_nodes: List[Node] = []
        for _ in range(c.N_REQUESTS):
            node_id = self._next_node_id
            self._next_node_id += 1

            t_i = np.random.uniform(0.0, c.DELTA_T)

            is_urgent = np.random.random() < c.URGENT_RATIO

            if is_urgent and c.URGENT_CORNER_RATIO > 0 and np.random.random() < c.URGENT_CORNER_RATIO:
                corner = random.choice(c.CORNER_POSITIONS)
                x = np.clip(np.random.normal(corner[0], c.URGENT_CORNER_SIGMA), 0, c.AREA_SIZE)
                y = np.clip(np.random.normal(corner[1], c.URGENT_CORNER_SIGMA), 0, c.AREA_SIZE)
            else:
                x = np.random.uniform(0, c.AREA_SIZE)
                y = np.random.uniform(0, c.AREA_SIZE)
            if is_urgent:
                node_type = 'urgent'
                demand = self._sample_truncated_demand(
                    c.URGENT_ENERGY_MIN, c.URGENT_ENERGY_MAX
                )
                tw_width = np.random.uniform(c.URGENT_TW_MIN, c.URGENT_TW_MAX)
            else:
                node_type = 'normal'
                demand = self._sample_truncated_demand(
                    c.NORMAL_ENERGY_MIN, c.NORMAL_ENERGY_MAX
                )
                tw_width = np.random.uniform(c.NORMAL_TW_MIN, c.NORMAL_TW_MAX)

            d_i = t_i + tw_width

            if is_urgent:
                ev_soc = np.random.uniform(c.URGENT_SOC_MIN, c.URGENT_SOC_MAX)
            else:
                ev_soc = max(0.05, 1.0 - demand / c.EV_BATTERY_CAPACITY)
            node = Node(
                id=node_id, x=x, y=y, demand=demand,
                ready_time=t_i, due_date=d_i, service_time=0,
                node_type=node_type, status='new',
                ev_current_soc=ev_soc,
            )
            new_nodes.append(node)
        return new_nodes

    def setup_nodes(self, customer_nodes: List[Node]) -> None:
        """設定節點列表並建立距離矩陣"""
        self.nodes = [self.depot] + customer_nodes
        self.dist_matrix = DistanceMatrix(self.nodes)

    # ==================== 距離 / 時間計算 ====================
    def calculate_distance(self, node1: Node, node2: Node, distance_type: str = 'euclidean') -> float:
        if self.dist_matrix is not None:
            vehicle = 'mcs_slow' if distance_type == 'manhattan' else 'uav'
            return self.dist_matrix.get_distance(node1, node2, vehicle)
        if distance_type == 'manhattan':
            return abs(node1.x - node2.x) + abs(node1.y - node2.y)
        else:
            return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

    def calculate_travel_time(self, node1: Node, node2: Node, vehicle: str = 'mcs_slow') -> float:
        c = self.cfg
        if self._is_mcs(vehicle):
            distance = self.calculate_distance(node1, node2, distance_type='manhattan')
            return distance / c.MCS_SPEED
        else:
            distance = self.calculate_distance(node1, node2, distance_type='euclidean')
            flight_time = distance / c.UAV_SPEED
            return flight_time + c.UAV_TAKEOFF_LAND_OVERHEAD if node1.id != node2.id else flight_time

    def calculate_charging_time(self, energy_kwh: float, power_kw: float) -> float:
        return (energy_kwh / power_kw) * 60.0

    def compute_uav_delivery(self, node: Node) -> float:
        """計算 UAV 實際交付電量：按需求充電，上限 UAV_DELIVERABLE_ENERGY"""
        c = self.cfg
        return min(node.demand, c.UAV_DELIVERABLE_ENERGY)

    def get_charging_profile(self, vehicle: str, node_type: str, demand: float,
                              node: Node = None) -> Tuple[str, float]:
        c = self.cfg
        if vehicle == 'uav':
            actual = self.compute_uav_delivery(node) if node is not None else min(demand, c.UAV_DELIVERABLE_ENERGY)
            return 'FAST', self.calculate_charging_time(actual, c.UAV_CHARGE_POWER)
        elif vehicle == 'mcs_fast':
            return 'FAST', self.calculate_charging_time(demand, c.MCS_FAST_POWER)
        else:
            return 'SLOW', self.calculate_charging_time(demand, c.MCS_SLOW_POWER)

    # ==================== 路徑評估 (含 depot-reachability) ====================
    def evaluate_route(self, route: Route) -> bool:
        """評估路徑可行性 — 含 depot-reachability energy constraint"""
        c = self.cfg
        if len(route.nodes) == 0:
            route.is_feasible = True
            route.total_distance = route.total_time = route.total_user_waiting_time = 0.0
            route.arrival_times, route.departure_times = [], []
            route.user_waiting_times, route.mcs_waiting_times, route.charging_modes = [], [], []
            return True

        vehicle = route.vehicle_type
        capacity = self._get_vehicle_capacity(vehicle)

        route.arrival_times, route.departure_times, route.charging_modes = [], [], []
        route.user_waiting_times, route.mcs_waiting_times = [], []
        route.total_distance = route.total_user_waiting_time = 0.0

        # 容量檢查: MCS 無限能量時跳過, UAV 用實際交付量
        if self._is_mcs(vehicle):
            if not c.MCS_UNLIMITED_ENERGY and route.total_demand > capacity:
                route.is_feasible = False
                return False
        else:
            uav_total = sum(self.compute_uav_delivery(n) for n in route.nodes)
            if uav_total > capacity:
                route.is_feasible = False
                return False

        # 起始狀態: 全部車輛 t=0 從 depot 出發、滿能量
        current_time = 0.0
        current_energy = capacity
        prev_node = self.depot

        setup, connect, disconnect = self._get_service_overhead(vehicle)

        for i, node in enumerate(route.nodes):
            travel_time = self.calculate_travel_time(prev_node, node, vehicle)
            dist_type = 'manhattan' if self._is_mcs(vehicle) else 'euclidean'
            distance = self.calculate_distance(prev_node, node, dist_type)

            route.total_distance += distance
            arrival_time = current_time + travel_time

            earliest_service = arrival_time + setup + connect
            service_start = max(earliest_service, node.ready_time)

            mode, charging_time = self.get_charging_profile(vehicle, node.node_type, node.demand, node=node)
            departure_time = service_start + charging_time + disconnect

            if departure_time > node.due_date:
                route.is_feasible = False
                return False

            user_waiting_time = max(0.0, arrival_time - node.ready_time)
            vehicle_idle_time = max(0.0, node.ready_time - earliest_service)

            # 能量扣減
            if self._is_mcs(vehicle):
                if not c.MCS_UNLIMITED_ENERGY:
                    travel_energy = distance * c.MCS_ENERGY_CONSUMPTION
                    current_energy -= (travel_energy + node.demand)
            else:
                # UAV: 扣減實際交付電量
                actual_delivery = self.compute_uav_delivery(node)
                current_energy -= actual_delivery
                if current_energy < 0:
                    route.is_feasible = False
                    return False

            route.arrival_times.append(arrival_time)
            route.departure_times.append(departure_time)
            route.charging_modes.append(mode)
            route.user_waiting_times.append(user_waiting_time)
            route.mcs_waiting_times.append(vehicle_idle_time)
            route.total_user_waiting_time += user_waiting_time

            current_time = departure_time
            prev_node = node

        # depot-reachability check: 完成 suffix 後是否能返回 depot
        if self._is_mcs(vehicle) and not c.MCS_UNLIMITED_ENERGY:
            last_node = route.nodes[-1]
            energy_to_depot = self._energy_to_depot(last_node)
            if current_energy < energy_to_depot:
                route.is_feasible = False
                return False

        # 回程距離 / 時間 (計入統計但不強制實際返回)
        dist_type = 'manhattan' if self._is_mcs(vehicle) else 'euclidean'
        return_dist = self.calculate_distance(prev_node, self.depot, dist_type)
        return_time = self.calculate_travel_time(prev_node, self.depot, vehicle)
        route.total_distance += return_dist
        route.total_time = current_time + return_time

        route.is_feasible = True
        return True

    # ==================== 插入啟發式 ====================
    def try_insert_node(self, route: Route, node: Node, position: int = None) -> bool:
        if position is not None:
            is_feasible, _ = self.incremental_insertion_check(route, position, node)
            if is_feasible:
                route.insert_node(position, node)
                self.evaluate_route(route)
                return True
            return False

        best_position = None
        best_delta_cost = float('inf')
        vehicle = route.vehicle_type
        dist_type = 'manhattan' if self._is_mcs(vehicle) else 'euclidean'

        for pos in range(len(route.nodes) + 1):
            is_feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
            if is_feasible:
                if delta_cost < best_delta_cost:
                    best_delta_cost = delta_cost
                    best_position = pos
                elif delta_cost == best_delta_cost and best_position is not None:
                    dist_pos = self._calculate_insertion_delta_dist(route, pos, node, dist_type)
                    dist_best = self._calculate_insertion_delta_dist(route, best_position, node, dist_type)
                    if dist_pos < dist_best:
                        best_position = pos

        if best_position is not None:
            route.insert_node(best_position, node)
            self.evaluate_route(route)
            return True
        return False

    def _calculate_insertion_delta_dist(self, route: Route, pos: int, node: Node, dist_type: str) -> float:
        prev_node = self.depot if pos == 0 else route.nodes[pos - 1]
        next_node = self.depot if pos == len(route.nodes) else route.nodes[pos]
        add_dist = (self.calculate_distance(prev_node, node, dist_type) +
                    self.calculate_distance(node, next_node, dist_type))
        sub_dist = self.calculate_distance(prev_node, next_node, dist_type)
        return add_dist - sub_dist

    def incremental_insertion_check(self, route: Route, pos: int, node: Node) -> Tuple[bool, float]:
        """增量插入可行性檢查，cost 採用增量 route time。"""
        vehicle = route.vehicle_type

        capacity = self._get_vehicle_capacity(vehicle)
        if vehicle == 'uav':
            delivery = self.compute_uav_delivery(node)
            if delivery <= 0:
                return False, float('inf')  # EV 已達安全水位
            uav_total = sum(self.compute_uav_delivery(n) for n in route.nodes) + delivery
            if uav_total > capacity:
                return False, float('inf')
        else:
            if not self.cfg.MCS_UNLIMITED_ENERGY and route.total_demand + node.demand > capacity:
                return False, float('inf')

        if len(route.departure_times) != len(route.nodes):
            if not self.evaluate_route(route):
                return False, float('inf')
        candidate = route.copy()
        candidate.insert_node(pos, node)
        if not self.evaluate_route(candidate):
            return False, float('inf')

        delta_cost = candidate.total_time - route.total_time
        return True, delta_cost

    def _forward_propagate_check(self, route: Route, start_pos: int,
                                  time_shift: float) -> Tuple[bool, float]:
        if time_shift <= 0:
            return True, 0.0

        delta_waiting = 0.0
        current_shift = time_shift
        vehicle = route.vehicle_type
        setup, connect, _ = self._get_service_overhead(vehicle)
        overhead = setup + connect

        for i in range(start_pos, len(route.nodes)):
            node = route.nodes[i]
            old_arrival = route.arrival_times[i]
            new_arrival = old_arrival + current_shift

            old_service_start = max(old_arrival + overhead, node.ready_time)
            new_service_start = max(new_arrival + overhead, node.ready_time)

            charging_time = route.departure_times[i] - old_service_start
            new_departure = new_service_start + charging_time

            if new_departure > node.due_date:
                return False, float('inf')

            old_waiting = route.user_waiting_times[i]
            new_waiting = max(0.0, new_arrival - node.ready_time)
            delta_waiting += (new_waiting - old_waiting)

            old_departure = route.departure_times[i]
            current_shift = new_departure - old_departure

            if current_shift <= 0:
                break

        return True, delta_waiting

    # ==================== Type Preference Bonus ====================
    def _get_type_preference_bonus(self, vehicle_type: str, node: Node) -> float:
        """
        計算車輛類型偏好獎勵 (分鐘)。用於 unified cost-based construction。

        策略：
        - MCS-SLOW 優先於 MCS-FAST（經濟性考量）
          bonus = 充電時間差，讓 Slow 的 adjusted_cost 與 Fast 對齊
        - UAV 僅用於 urgent 且 slack 極緊的請求
        """
        c = self.cfg
        if vehicle_type == 'mcs_slow':
            slow_time = (node.demand / c.MCS_SLOW_POWER) * 60.0
            fast_time = (node.demand / c.MCS_FAST_POWER) * 60.0
            return slow_time - fast_time
        elif vehicle_type == 'uav':
            if node.node_type == 'urgent':
                slack = node.due_date - node.ready_time
                if slack <= 15.0:
                    return c.UAV_URGENT_BONUS
            return 0.0
        else:  # mcs_fast
            return 0.0

    # ==================== 建構啟發式 (Type-Flexible Unified Cost) ====================
    def greedy_insertion_construction(
        self,
        active_requests: List[Node],
        vehicle_states: List[VehicleState],
    ) -> Solution:
        """
        Type-Flexible Greedy Insertion Construction

        對每個請求 (按 due_date 排序):
          1. 同時搜尋所有車隊 (MCS-SLOW, MCS-FAST, UAV) 的最佳插入位置
          2. 使用 soft type preference bonus 取代 hard priority
          3. 選擇 adjusted_cost 最小的全局最佳插入
          4. Reserve activation (for remaining unassigned)
        """
        c = self.cfg
        solution = Solution(c)

        for vs in vehicle_states:
            route = Route(vehicle_type=vs.vehicle_type, vehicle_id=vs.vehicle_id)
            if vs.vehicle_type == 'uav':
                solution.add_uav_route(route)
            else:
                solution.add_mcs_route(route)
            self.evaluate_route(route)

        # 排序 — 依策略決定 request 處理順序
        strategy = c.CONSTRUCTION_STRATEGY
        if strategy == "edf":
            # Earliest Deadline First — 只看 due_date，不考慮空間/服務時間
            sorted_requests = sorted(
                active_requests,
                key=lambda n: (n.due_date, -n.demand)
            )
        elif strategy == "slack":
            # Slack-based — 考慮空間可達性與服務時間 (static: 全部車輛在 depot)
            active_positions = [vs.position for vs in vehicle_states]
            max_power = max(c.MCS_FAST_POWER, c.MCS_SLOW_POWER)

            def _slack(n: Node) -> float:
                if active_positions:
                    min_travel = min(
                        (abs(n.x - p.x) + abs(n.y - p.y)) / c.MCS_SPEED
                        for p in active_positions
                    )
                else:
                    min_travel = (abs(n.x - self.depot.x) + abs(n.y - self.depot.y)) / c.MCS_SPEED
                min_service = (n.demand / max_power) * 60.0
                return n.due_date - (min_travel + min_service)

            sorted_requests = sorted(active_requests, key=lambda n: (_slack(n), n.due_date))
        elif strategy == "nearest":
            sorted_requests = sorted(
                active_requests,
                key=lambda n: abs(n.x - self.depot.x) + abs(n.y - self.depot.y)
            )
        elif strategy == "regret2":
            # regret2 走另一條入口 (regret2_insertion_construction)；
            # 直接呼叫 greedy 時 fallback 為 due_date 排序
            sorted_requests = sorted(active_requests, key=lambda n: n.due_date)
        else:
            raise ValueError(
                f"Unknown CONSTRUCTION_STRATEGY: {strategy!r} "
                f"(expected one of 'edf', 'slack', 'nearest', 'regret2')"
            )
        unassigned: List[Node] = []

        # Type-Flexible Unified Cost Dispatching
        for node in sorted_requests:
            best_route_type: Optional[str] = None
            best_route_idx = -1
            best_position = -1
            min_adjusted_cost = float('inf')

            # --- Search ALL MCS routes (SLOW + FAST) simultaneously ---
            for r_idx, route in enumerate(solution.mcs_routes):
                for pos in range(len(route.nodes) + 1):
                    feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
                    if feasible:
                        bonus = self._get_type_preference_bonus(route.vehicle_type, node)
                        adjusted_cost = delta_cost - bonus
                        if adjusted_cost < min_adjusted_cost:
                            min_adjusted_cost = adjusted_cost
                            best_route_type = 'mcs'
                            best_route_idx = r_idx
                            best_position = pos

            # --- Search UAV routes — 僅在所有 MCS 都不可行時 ---
            if (best_route_type is None
                    and node.node_type == 'urgent'
                    and self.compute_uav_delivery(node) > 0):
                for r_idx, route in enumerate(solution.uav_routes):
                    for pos in range(len(route.nodes) + 1):
                        feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
                        if feasible:
                            bonus = self._get_type_preference_bonus('uav', node)
                            adjusted_cost = delta_cost - bonus
                            if adjusted_cost < min_adjusted_cost:
                                min_adjusted_cost = adjusted_cost
                                best_route_type = 'uav'
                                best_route_idx = r_idx
                                best_position = pos

            # 執行插入
            inserted = False
            if best_route_type == 'mcs':
                target = solution.mcs_routes[best_route_idx]
                target.insert_node(best_position, node)
                if self.evaluate_route(target):
                    node.status = 'assigned'
                    inserted = True
                else:
                    target.remove_node(best_position)
                    self.evaluate_route(target)

            elif best_route_type == 'uav':
                target = solution.uav_routes[best_route_idx]
                target.insert_node(best_position, node)
                if self.evaluate_route(target):
                    node.status = 'assigned'
                    inserted = True
                else:
                    target.remove_node(best_position)
                    self.evaluate_route(target)

            if not inserted:
                unassigned.append(node)

        solution.unassigned_nodes = unassigned
        solution.calculate_total_cost(len(active_requests))
        return solution

    # ==================== Regret-2 Insertion Construction ====================
    def regret2_insertion_construction(
        self,
        active_requests: List[Node],
        vehicle_states: List[VehicleState],
    ) -> Solution:
        """
        Priority-Class Regret-2 Insertion Construction

        Phase 1: 對 urgent 節點跑 Regret-2 — 確保 urgent 優先佔位
        Phase 2: 對 normal 節點跑 Regret-2 — 在剩餘空間中最佳化
        """
        c = self.cfg
        solution = Solution(c)

        for vs in vehicle_states:
            route = Route(vehicle_type=vs.vehicle_type, vehicle_id=vs.vehicle_id)
            if vs.vehicle_type == 'uav':
                solution.add_uav_route(route)
            else:
                solution.add_mcs_route(route)
            self.evaluate_route(route)

        # Priority-Class: urgent 先, normal 後
        urgent_pool = [n for n in active_requests if n.node_type == 'urgent']
        normal_pool = [n for n in active_requests if n.node_type == 'normal']

        unassigned: List[Node] = []
        unassigned += self._regret2_insert_pool(solution, urgent_pool)
        unassigned += self._regret2_insert_pool(solution, normal_pool)

        solution.unassigned_nodes = unassigned
        solution.calculate_total_cost(len(active_requests))
        return solution

    def _regret2_insert_pool(
        self,
        solution: Solution,
        pool: List[Node],
    ) -> List[Node]:
        """Regret-2 迴圈：對給定 pool 中的節點依 regret 動態排序插入，回傳未能插入的節點。"""
        pool = list(pool)
        unassigned: List[Node] = []

        while pool:
            best_priority = None  # (regret, -due_date, -best_cost)
            best_node = None
            best_node_idx = -1
            best_insertion = None

            for n_idx, node in enumerate(pool):
                options: List[Tuple[float, str, int, int]] = []

                # 搜尋所有 MCS 路徑
                for r_idx, route in enumerate(solution.mcs_routes):
                    for pos in range(len(route.nodes) + 1):
                        feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
                        if feasible:
                            bonus = self._get_type_preference_bonus(route.vehicle_type, node)
                            adjusted_cost = delta_cost - bonus
                            options.append((adjusted_cost, 'mcs', r_idx, pos))

                # 僅當 MCS 全部 infeasible 且為 urgent 時搜尋 UAV
                if (not options
                        and node.node_type == 'urgent'
                        and self.compute_uav_delivery(node) > 0):
                    for r_idx, route in enumerate(solution.uav_routes):
                        for pos in range(len(route.nodes) + 1):
                            feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
                            if feasible:
                                bonus = self._get_type_preference_bonus('uav', node)
                                adjusted_cost = delta_cost - bonus
                                options.append((adjusted_cost, 'uav', r_idx, pos))

                if not options:
                    continue

                options.sort(key=lambda x: x[0])
                best_cost = options[0][0]
                second_best_cost = options[1][0] if len(options) > 1 else float('inf')
                regret = second_best_cost - best_cost

                priority = (regret, -node.due_date, -best_cost)

                if best_priority is None or priority > best_priority:
                    best_priority = priority
                    best_node = node
                    best_node_idx = n_idx
                    best_insertion = (options[0][1], options[0][2], options[0][3], best_cost)

            if best_node is None:
                for node in pool:
                    unassigned.append(node)
                pool.clear()
                break

            rt, ri, rp, _ = best_insertion
            inserted = False
            if rt == 'mcs':
                target = solution.mcs_routes[ri]
            else:
                target = solution.uav_routes[ri]

            target.insert_node(rp, best_node)
            if self.evaluate_route(target):
                best_node.status = 'assigned'
                inserted = True
            else:
                target.remove_node(rp)
                self.evaluate_route(target)

            if not inserted:
                unassigned.append(best_node)

            pool.pop(best_node_idx)

        return unassigned

    # ==================== Nearest-Neighbor Chain Construction ====================
    def nearest_neighbor_construction(
        self,
        active_requests: List[Node],
        vehicle_states: List[VehicleState],
    ) -> Solution:
        """
        Nearest-Neighbor Chain — 從車輛當前位置出發，每次選距離最近的可行請求加到路徑尾端。

        車輛優先順序: MCS-SLOW → MCS-FAST → UAV
        每台車從自身位置出發，不斷挑最近且可行的請求串接，直到沒有可行請求為止。
        """
        c = self.cfg
        solution = Solution(c)

        for vs in vehicle_states:
            route = Route(vehicle_type=vs.vehicle_type, vehicle_id=vs.vehicle_id)
            if vs.vehicle_type == 'uav':
                solution.add_uav_route(route)
            else:
                solution.add_mcs_route(route)
            self.evaluate_route(route)

        remaining = list(active_requests)

        # --- Phase 1: MCS-SLOW vehicles ---
        for route in solution.mcs_routes:
            if route.vehicle_type != 'mcs_slow':
                continue
            self._nn_fill_route(route, remaining)

        # --- Phase 2: MCS-FAST vehicles ---
        for route in solution.mcs_routes:
            if route.vehicle_type != 'mcs_fast':
                continue
            self._nn_fill_route(route, remaining)

        # --- Phase 3: UAV vehicles ---
        for route in solution.uav_routes:
            uav_remaining = [n for n in remaining
                             if n.node_type == 'urgent' and self.compute_uav_delivery(n) > 0]
            self._nn_fill_route(route, uav_remaining)
            assigned_ids = {n.id for n in route.nodes}
            remaining = [n for n in remaining if n.id not in assigned_ids]

        for node in remaining:
            node.status = 'unassigned'
        solution.unassigned_nodes = remaining
        solution.calculate_total_cost(len(active_requests))
        return solution

    def _nn_fill_route(self, route: Route, remaining: List[Node]) -> None:
        """對單一路徑，反覆從尾端挑最近可行請求串接。"""
        while remaining:
            # 當前位置 = 路徑最後一個 node，否則 depot
            current = route.nodes[-1] if route.nodes else self.depot

            # 按距離排序，找最近的可行請求
            remaining.sort(key=lambda n: self._nn_distance(current, n, route.vehicle_type))

            inserted = False
            for i, node in enumerate(remaining):
                # 嘗試插入到尾端
                pos = len(route.nodes)
                route.insert_node(pos, node)
                if self.evaluate_route(route):
                    node.status = 'assigned'
                    remaining.pop(i)
                    inserted = True
                    break
                else:
                    route.remove_node(pos)
                    self.evaluate_route(route)

            if not inserted:
                break

    def _nn_distance(self, from_node: Node, to_node: Node, vehicle_type: str) -> float:
        """計算兩節點間的距離 (MCS=曼哈頓, UAV=歐式)"""
        if self._is_mcs(vehicle_type):
            return abs(from_node.x - to_node.x) + abs(from_node.y - to_node.y)
        else:
            return np.sqrt((from_node.x - to_node.x)**2 + (from_node.y - to_node.y)**2)

    def print_config(self) -> None:
        c = self.cfg
        print("=" * 80)
        print("Experiment Config (Static Single-Batch)")
        print("=" * 80)
        print(f"  N_REQUESTS = {c.N_REQUESTS}, urgent_ratio = {c.URGENT_RATIO}")
        print(f"  Construction strategy: {c.CONSTRUCTION_STRATEGY}")
        print(f"  Fleet: S-MCS={c.NUM_MCS_SLOW}, F-MCS={c.NUM_MCS_FAST}, UAV={c.NUM_UAV}")
        print(f"  MCS: {c.MCS_SPEED * 60:.1f}km/h, {c.MCS_CAPACITY}kWh, "
              f"Slow={c.MCS_SLOW_POWER}kW, Fast={c.MCS_FAST_POWER}kW")
        print(f"  UAV: {c.UAV_SPEED * 60:.1f}km/h, {c.UAV_DELIVERABLE_ENERGY}kWh, "
              f"{c.UAV_CHARGE_POWER}kW")
        print("=" * 80 + "\n")
