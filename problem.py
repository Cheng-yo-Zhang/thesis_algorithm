import numpy as np
import random
from typing import List, Set, Tuple, Dict, Optional

from config import Config
from models import Node, VehicleState, Route, Solution, DistanceMatrix


class ChargingSchedulingProblem:
    """充電排程問題 — 批次調度 (Batch Dispatching)"""

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        c = self.cfg

        self.nodes: List[Node] = []
        self.depot: Node = None
        self.uav_eligible_ids: Set[int] = set()
        self.dist_matrix: Optional[DistanceMatrix] = None
        self._next_node_id: int = 1

        random.seed(c.RANDOM_SEED)
        np.random.seed(c.RANDOM_SEED)

        self.depot = Node(
            id=0, x=c.DEPOT_X, y=c.DEPOT_Y,
            demand=0, ready_time=0, due_date=c.T_TOTAL,
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

    # ==================== 需求生成 (HPP / NHPP) ====================
    def generate_requests(self, slot_start: float, slot_number: int = 0) -> List[Node]:
        """依 HPP 或 NHPP 生成單一排程週期的充電請求"""
        c = self.cfg

        # 超過 GENERATION_END 就不再生成新請求 (cooldown phase)
        if c.GENERATION_END > 0 and slot_start >= c.GENERATION_END:
            return []

        r_t = c.get_arrival_rate_profile(slot_start)
        mu_k = c.LAMBDA_BAR * r_t * c.DELTA_T
        if c.USE_FIXED_DEMAND:
            n_requests = c.FIXED_DEMAND_PER_SLOT
        else:
            n_requests = np.random.poisson(mu_k)

        new_nodes: List[Node] = []
        for _ in range(n_requests):
            node_id = self._next_node_id
            self._next_node_id += 1

            t_i = np.random.uniform(slot_start, slot_start + c.DELTA_T)

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
                ev_current_soc=ev_soc, original_demand=demand,
                uav_served=False, residual_demand=demand,
                origin_slot=slot_number
            )
            new_nodes.append(node)
        return new_nodes

    def setup_nodes(self, customer_nodes: List[Node]) -> None:
        """設定節點列表並建立距離矩陣"""
        self.nodes = [self.depot] + customer_nodes
        self.dist_matrix = DistanceMatrix(self.nodes)
        self._compute_uav_eligible()

    def _compute_uav_eligible(self) -> None:
        """所有客戶皆可由 UAV 服務 (距離限制已取消)"""
        customers = [n for n in self.nodes if n.node_type != 'depot']
        self.uav_eligible_ids = {c.id for c in customers}

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

        # 起始狀態: 從 release state 出發
        start_node = route.start_position if route.start_position else self.depot
        current_time = route.start_time
        current_energy = route.start_energy if route.start_energy > 0 else capacity
        prev_node = start_node

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
        start_for_route = route.start_position if route.start_position else self.depot
        prev_node = start_for_route if pos == 0 else route.nodes[pos - 1]
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
            # 已被 UAV 服務過的節點不再由 UAV 服務
            if node.uav_served:
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
    def _get_type_preference_bonus(self, vehicle_type: str, node: Node,
                                     dispatch_window_end: float = None) -> float:
        """
        計算車輛類型偏好獎勵 (分鐘)。用於 unified cost-based construction。

        策略：
        - MCS-SLOW 優先於 MCS-FAST（經濟性考量）
          bonus = 充電時間差，讓 Slow 的 adjusted_cost 與 Fast 對齊
        - UAV 僅用於 urgent 且 slack 極緊的請求
        """
        c = self.cfg
        if vehicle_type == 'mcs_slow':
            # bonus = Fast 與 Slow 的充電時間差，讓兩者在 adjusted_cost 上對齊
            # 這樣只要 Slow 能在時間窗內完成，就會優先於 Fast
            slow_time = (node.demand / c.MCS_SLOW_POWER) * 60.0  # min
            fast_time = (node.demand / c.MCS_FAST_POWER) * 60.0  # min
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
        dispatch_window_end: float = None
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

        # 為每台 active vehicle 建立 route (suffix)
        vehicle_route_map: Dict[int, Tuple[str, int]] = {}

        for vs in vehicle_states:
            if not vs.is_active:
                continue
            route = Route(vehicle_type=vs.vehicle_type, vehicle_id=vs.vehicle_id)
            route.start_position = vs.position
            route.start_time = vs.available_time
            route.start_energy = vs.remaining_energy
            route.is_deployed = len(vs.committed_nodes) > 0

            if vs.vehicle_type == 'uav':
                solution.add_uav_route(route)
                vehicle_route_map[vs.vehicle_id] = ('uav', len(solution.uav_routes) - 1)
            else:
                solution.add_mcs_route(route)
                vehicle_route_map[vs.vehicle_id] = ('mcs', len(solution.mcs_routes) - 1)

            self.evaluate_route(route)

        # 排序 — 依策略決定 request 處理順序
        if c.CONSTRUCTION_STRATEGY == "edf":
            # Earliest Deadline First — 只看 due_date，不考慮空間/服務時間
            sorted_requests = sorted(
                active_requests,
                key=lambda n: (n.due_date, -n.demand)
            )
        elif c.CONSTRUCTION_STRATEGY == "slack":
            # Slack-based — 考慮空間可達性與服務時間
            dispatch_time = (dispatch_window_end - c.DELTA_T) if dispatch_window_end else 0.0
            active_positions = [vs.position for vs in vehicle_states if vs.is_active]
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
                return n.due_date - (dispatch_time + min_travel + min_service)

            sorted_requests = sorted(active_requests, key=lambda n: (_slack(n), n.due_date))
        elif c.CONSTRUCTION_STRATEGY == "nearest":
            sorted_requests = sorted(
                active_requests,
                key=lambda n: abs(n.x - self.depot.x) + abs(n.y - self.depot.y)
            )
        else:
            # fallback: due_date
            sorted_requests = sorted(active_requests, key=lambda n: n.due_date)
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
                        bonus = self._get_type_preference_bonus(route.vehicle_type, node, dispatch_window_end)
                        adjusted_cost = delta_cost - bonus
                        if adjusted_cost < min_adjusted_cost:
                            min_adjusted_cost = adjusted_cost
                            best_route_type = 'mcs'
                            best_route_idx = r_idx
                            best_position = pos

            # --- Search UAV routes — 僅在所有 MCS 都不可行時 ---
            if (best_route_type is None
                    and node.node_type == 'urgent'
                    and self.compute_uav_delivery(node) > 0
                    and not node.uav_served):
                for r_idx, route in enumerate(solution.uav_routes):
                    for pos in range(len(route.nodes) + 1):
                        feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
                        if feasible:
                            bonus = self._get_type_preference_bonus('uav', node, dispatch_window_end)
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

        # Phase 2: Reserve MCS Activation — Section 6.3
        if unassigned and c.ENABLE_RESERVE_ACTIVATION:
            unassigned = self._try_reserve_activation(
                solution, unassigned, vehicle_states
            )

        solution.unassigned_nodes = unassigned
        solution.calculate_total_cost(len(active_requests))
        return solution

    # ==================== Regret-2 Insertion Construction ====================
    def regret2_insertion_construction(
        self,
        active_requests: List[Node],
        vehicle_states: List[VehicleState],
        dispatch_window_end: float = None
    ) -> Solution:
        """
        Priority-Class Regret-2 Insertion Construction

        Phase 1: 對 urgent 節點跑 Regret-2 — 確保 urgent 優先佔位
        Phase 2: 對 normal 節點跑 Regret-2 — 在剩餘空間中最佳化
        """
        c = self.cfg
        solution = Solution(c)

        # 為每台 active vehicle 建立 route (suffix) — 與 greedy 相同
        vehicle_route_map: Dict[int, Tuple[str, int]] = {}

        for vs in vehicle_states:
            if not vs.is_active:
                continue
            route = Route(vehicle_type=vs.vehicle_type, vehicle_id=vs.vehicle_id)
            route.start_position = vs.position
            route.start_time = vs.available_time
            route.start_energy = vs.remaining_energy
            route.is_deployed = len(vs.committed_nodes) > 0

            if vs.vehicle_type == 'uav':
                solution.add_uav_route(route)
                vehicle_route_map[vs.vehicle_id] = ('uav', len(solution.uav_routes) - 1)
            else:
                solution.add_mcs_route(route)
                vehicle_route_map[vs.vehicle_id] = ('mcs', len(solution.mcs_routes) - 1)

            self.evaluate_route(route)

        # Priority-Class: urgent 先, normal 後
        urgent_pool = [n for n in active_requests if n.node_type == 'urgent']
        normal_pool = [n for n in active_requests if n.node_type == 'normal']

        unassigned: List[Node] = []
        unassigned += self._regret2_insert_pool(solution, urgent_pool, dispatch_window_end)
        unassigned += self._regret2_insert_pool(solution, normal_pool, dispatch_window_end)

        # Reserve MCS Activation
        if unassigned and c.ENABLE_RESERVE_ACTIVATION:
            unassigned = self._try_reserve_activation(
                solution, unassigned, vehicle_states
            )

        solution.unassigned_nodes = unassigned
        solution.calculate_total_cost(len(active_requests))
        return solution

    def _regret2_insert_pool(
        self,
        solution: Solution,
        pool: List[Node],
        dispatch_window_end: float = None
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
                            bonus = self._get_type_preference_bonus(route.vehicle_type, node, dispatch_window_end)
                            adjusted_cost = delta_cost - bonus
                            options.append((adjusted_cost, 'mcs', r_idx, pos))

                # 僅當 MCS 全部 infeasible 且為 urgent 時搜尋 UAV
                if (not options
                        and node.node_type == 'urgent'
                        and self.compute_uav_delivery(node) > 0
                        and not node.uav_served):
                    for r_idx, route in enumerate(solution.uav_routes):
                        for pos in range(len(route.nodes) + 1):
                            feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
                            if feasible:
                                bonus = self._get_type_preference_bonus('uav', node, dispatch_window_end)
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

    # ==================== Sweep + Regret-2 Construction ====================
    # ==================== Clarke-Wright Savings Construction ====================
    def savings_construction(
        self,
        active_requests: List[Node],
        vehicle_states: List[VehicleState],
        dispatch_window_end: float = None
    ) -> Solution:
        """
        Clarke-Wright Savings Algorithm (adapted for VRPTW + multi-type fleet)

        Phase 1: 每個客戶一條虛擬路徑
        Phase 2: 計算 savings 並由大到小合併
        Phase 3: 分配車型 (SLOW 優先, FAST 補位)
        Phase 4: UAV fallback (urgent only)
        """
        c = self.cfg
        solution = Solution(c)

        mcs_states = [vs for vs in vehicle_states if vs.is_active and vs.vehicle_type != 'uav']
        uav_states = [vs for vs in vehicle_states if vs.is_active and vs.vehicle_type == 'uav']
        n_mcs = len(mcs_states)

        # 建立 UAV 空路徑
        for vs in uav_states:
            route = Route(vehicle_type=vs.vehicle_type, vehicle_id=vs.vehicle_id)
            route.start_position = vs.position
            route.start_time = vs.available_time
            route.start_energy = vs.remaining_energy
            solution.add_uav_route(route)
            self.evaluate_route(route)

        if n_mcs == 0 or not active_requests:
            solution.unassigned_nodes = list(active_requests)
            solution.calculate_total_cost(len(active_requests))
            return solution

        depot = self.depot
        requests = list(active_requests)

        # --- Phase 1: 每個客戶一條虛擬路徑 (chain list) ---
        # route_of[node.id] = route_index
        # routes = list of lists of nodes
        routes = [[node] for node in requests]
        route_of = {}
        for r_idx, r in enumerate(routes):
            route_of[r[0].id] = r_idx

        # --- Phase 2: 計算 savings ---
        savings_list = []
        for i, ni in enumerate(requests):
            d_depot_i = abs(depot.x - ni.x) + abs(depot.y - ni.y)  # manhattan
            for j, nj in enumerate(requests):
                if i >= j:
                    continue
                d_depot_j = abs(depot.x - nj.x) + abs(depot.y - nj.y)
                d_ij = abs(ni.x - nj.x) + abs(ni.y - nj.y)
                sav = d_depot_i + d_depot_j - d_ij
                if sav > 0:
                    savings_list.append((sav, ni.id, nj.id))

        savings_list.sort(key=lambda x: -x[0])

        # --- Phase 3: 合併路徑 ---
        # 合併條件: i 在某路徑尾端, j 在另一路徑首端 (或反過來), 合併後可行
        for sav, id_i, id_j in savings_list:
            ri = route_of.get(id_i)
            rj = route_of.get(id_j)
            if ri is None or rj is None or ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]
            if route_i is None or route_j is None:
                continue

            # 嘗試合併: route_i + route_j (i 在尾, j 在首)
            merged = None
            if route_i[-1].id == id_i and route_j[0].id == id_j:
                merged = route_i + route_j
            elif route_j[-1].id == id_j and route_i[0].id == id_i:
                merged = route_j + route_i
            elif route_i[-1].id == id_i and route_j[-1].id == id_j:
                merged = route_i + route_j[::-1]
            elif route_i[0].id == id_i and route_j[0].id == id_j:
                merged = route_i[::-1] + route_j

            if merged is None:
                continue

            # 檢查合併後是否可行 (用最快車型 MCS-FAST 檢查上界)
            test_route = Route(vehicle_type='mcs_fast', vehicle_id=0)
            test_route.start_position = depot
            test_route.start_time = 0.0
            test_route.start_energy = c.MCS_CAPACITY
            test_route.nodes = list(merged)
            test_route.total_demand = sum(n.demand for n in merged)

            if self.evaluate_route(test_route):
                # 合併可行
                routes[ri] = merged
                routes[rj] = None
                for node in merged:
                    route_of[node.id] = ri
            # 不可行就跳過, 不合併

        # 收集非空路徑
        valid_routes = [r for r in routes if r is not None]

        # 按節點數由多到少排序 (大路徑優先分配)
        valid_routes.sort(key=lambda r: -len(r))

        # --- Phase 4: 分配到實際 MCS 車輛 ---
        # SLOW 優先, 時間窗不夠才用 FAST
        slow_states = [vs for vs in mcs_states if vs.vehicle_type == 'mcs_slow']
        fast_states = [vs for vs in mcs_states if vs.vehicle_type == 'mcs_fast']
        ordered_states = slow_states + fast_states

        assigned_nodes = set()
        for vs in ordered_states:
            route = Route(vehicle_type=vs.vehicle_type, vehicle_id=vs.vehicle_id)
            route.start_position = vs.position
            route.start_time = vs.available_time
            route.start_energy = vs.remaining_energy
            solution.add_mcs_route(route)
            self.evaluate_route(route)

        # 貪婪分配: 每條合併路徑嘗試放入最適合的 MCS
        for chain in valid_routes:
            best_mcs_idx = -1
            best_cost = float('inf')

            for m_idx, mcs_route in enumerate(solution.mcs_routes):
                if len(mcs_route.nodes) > 0:
                    continue  # 已被佔用

                # 嘗試整條 chain 放入
                test = mcs_route.copy()
                test.nodes = list(chain)
                test.total_demand = sum(n.demand for n in chain)
                if self.evaluate_route(test):
                    if test.total_time < best_cost:
                        best_cost = test.total_time
                        best_mcs_idx = m_idx

            if best_mcs_idx >= 0:
                target = solution.mcs_routes[best_mcs_idx]
                target.nodes = list(chain)
                target.total_demand = sum(n.demand for n in chain)
                self.evaluate_route(target)
                for node in chain:
                    node.status = 'assigned'
                    assigned_nodes.add(node.id)
            else:
                # 整條 chain 放不進任何空車 → 逐一插入
                for node in chain:
                    if node.id in assigned_nodes:
                        continue
                    inserted = False
                    best_r = -1
                    best_p = -1
                    min_cost = float('inf')
                    for m_idx, mcs_route in enumerate(solution.mcs_routes):
                        for pos in range(len(mcs_route.nodes) + 1):
                            feasible, delta = self.incremental_insertion_check(mcs_route, pos, node)
                            if feasible:
                                bonus = self._get_type_preference_bonus(mcs_route.vehicle_type, node, dispatch_window_end)
                                adj = delta - bonus
                                if adj < min_cost:
                                    min_cost = adj
                                    best_r = m_idx
                                    best_p = pos

                    if best_r >= 0:
                        target = solution.mcs_routes[best_r]
                        target.insert_node(best_p, node)
                        if self.evaluate_route(target):
                            node.status = 'assigned'
                            assigned_nodes.add(node.id)
                            inserted = True
                        else:
                            target.remove_node(best_p)
                            self.evaluate_route(target)

        # 收集未分配的
        unassigned = [n for n in requests if n.id not in assigned_nodes]

        # 未分配的再做一輪全域 greedy insertion
        still_unassigned = []
        for node in unassigned:
            best_r = -1
            best_p = -1
            min_cost = float('inf')
            for m_idx, mcs_route in enumerate(solution.mcs_routes):
                for pos in range(len(mcs_route.nodes) + 1):
                    feasible, delta = self.incremental_insertion_check(mcs_route, pos, node)
                    if feasible:
                        bonus = self._get_type_preference_bonus(mcs_route.vehicle_type, node, dispatch_window_end)
                        adj = delta - bonus
                        if adj < min_cost:
                            min_cost = adj
                            best_r = m_idx
                            best_p = pos

            if best_r >= 0:
                target = solution.mcs_routes[best_r]
                target.insert_node(best_p, node)
                if self.evaluate_route(target):
                    node.status = 'assigned'
                else:
                    target.remove_node(best_p)
                    self.evaluate_route(target)
                    still_unassigned.append(node)
            else:
                still_unassigned.append(node)

        # --- Phase 5: UAV fallback (urgent only) ---
        final_unassigned = []
        for node in still_unassigned:
            if (node.node_type == 'urgent'
                    and self.compute_uav_delivery(node) > 0
                    and not node.uav_served):
                inserted = False
                for route in solution.uav_routes:
                    for pos in range(len(route.nodes) + 1):
                        feasible, delta = self.incremental_insertion_check(route, pos, node)
                        if feasible:
                            route.insert_node(pos, node)
                            if self.evaluate_route(route):
                                node.status = 'assigned'
                                inserted = True
                                break
                            else:
                                route.remove_node(pos)
                                self.evaluate_route(route)
                    if inserted:
                        break
                if not inserted:
                    final_unassigned.append(node)
            else:
                final_unassigned.append(node)

        solution.unassigned_nodes = final_unassigned
        solution.calculate_total_cost(len(active_requests))
        return solution

    # ==================== Sweep + Regret-2 Construction ====================
    def sweep_regret2_construction(
        self,
        active_requests: List[Node],
        vehicle_states: List[VehicleState],
        dispatch_window_end: float = None
    ) -> Solution:
        """
        Sweep + Regret-2 Construction

        Phase 1: 以 depot 為圓心，按極角將客戶分配到 MCS 扇區
        Phase 2: 每個扇區內用 Regret-2 決定插入順序
        Phase 3: MCS 無法服務的 urgent → UAV fallback
        """
        import math
        c = self.cfg
        solution = Solution(c)

        # --- 建立空路徑 ---
        mcs_states = [vs for vs in vehicle_states if vs.is_active and vs.vehicle_type != 'uav']
        uav_states = [vs for vs in vehicle_states if vs.is_active and vs.vehicle_type == 'uav']

        for vs in mcs_states:
            route = Route(vehicle_type=vs.vehicle_type, vehicle_id=vs.vehicle_id)
            route.start_position = vs.position
            route.start_time = vs.available_time
            route.start_energy = vs.remaining_energy
            route.is_deployed = len(vs.committed_nodes) > 0
            solution.add_mcs_route(route)
            self.evaluate_route(route)

        for vs in uav_states:
            route = Route(vehicle_type=vs.vehicle_type, vehicle_id=vs.vehicle_id)
            route.start_position = vs.position
            route.start_time = vs.available_time
            route.start_energy = vs.remaining_energy
            route.is_deployed = len(vs.committed_nodes) > 0
            solution.add_uav_route(route)
            self.evaluate_route(route)

        n_mcs = len(solution.mcs_routes)
        if n_mcs == 0:
            solution.unassigned_nodes = list(active_requests)
            solution.calculate_total_cost(len(active_requests))
            return solution

        # --- Phase 1: Sweep 分區 ---
        depot_x, depot_y = self.depot.x, self.depot.y

        # 計算極角並排序
        def polar_angle(node):
            return math.atan2(node.y - depot_y, node.x - depot_x)

        sorted_by_angle = sorted(active_requests, key=polar_angle)

        # 均分到 MCS 扇區
        clusters = [[] for _ in range(n_mcs)]
        for i, node in enumerate(sorted_by_angle):
            clusters[i % n_mcs].append(node)

        # --- Phase 2: 每個扇區內用 Regret-2 插入 (優先插入本扇區路徑) ---
        spillover = []  # 本扇區塞不下的節點
        for cluster_idx, cluster_nodes in enumerate(clusters):
            route = solution.mcs_routes[cluster_idx]

            urgent_nodes = [n for n in cluster_nodes if n.node_type == 'urgent']
            normal_nodes = [n for n in cluster_nodes if n.node_type == 'normal']

            for pool in [urgent_nodes, normal_nodes]:
                remaining = list(pool)
                while remaining:
                    best_node = None
                    best_node_idx = -1
                    best_pos = -1
                    best_priority = None

                    for n_idx, node in enumerate(remaining):
                        options = []
                        for pos in range(len(route.nodes) + 1):
                            feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
                            if feasible:
                                bonus = self._get_type_preference_bonus(route.vehicle_type, node, dispatch_window_end)
                                options.append((delta_cost - bonus, pos))

                        if not options:
                            continue

                        options.sort(key=lambda x: x[0])
                        best_cost = options[0][0]
                        second_best = options[1][0] if len(options) > 1 else float('inf')
                        regret = second_best - best_cost
                        priority = (regret, -node.due_date, -best_cost)

                        if best_priority is None or priority > best_priority:
                            best_priority = priority
                            best_node = node
                            best_node_idx = n_idx
                            best_pos = options[0][1]

                    if best_node is None:
                        spillover.extend(remaining)
                        break

                    route.insert_node(best_pos, best_node)
                    if self.evaluate_route(route):
                        best_node.status = 'assigned'
                    else:
                        route.remove_node(best_pos)
                        self.evaluate_route(route)
                        spillover.append(best_node)

                    remaining.pop(best_node_idx)

        # --- Phase 2b: 跨區 Regret-2 — 溢出節點搜尋所有 MCS 路徑 ---
        unassigned = self._regret2_insert_pool(solution, spillover, dispatch_window_end)

        # --- Phase 3: UAV fallback (urgent only) ---
        still_unassigned = []
        for node in unassigned:
            if (node.node_type == 'urgent'
                    and self.compute_uav_delivery(node) > 0
                    and not node.uav_served):
                inserted = False
                for r_idx, route in enumerate(solution.uav_routes):
                    for pos in range(len(route.nodes) + 1):
                        feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
                        if feasible:
                            route.insert_node(pos, node)
                            if self.evaluate_route(route):
                                node.status = 'assigned'
                                inserted = True
                                break
                            else:
                                route.remove_node(pos)
                                self.evaluate_route(route)
                    if inserted:
                        break
                if not inserted:
                    still_unassigned.append(node)
            else:
                still_unassigned.append(node)

        solution.unassigned_nodes = still_unassigned
        solution.calculate_total_cost(len(active_requests))
        return solution

    # ==================== Nearest-Neighbor Chain Construction ====================
    def nearest_neighbor_construction(
        self,
        active_requests: List[Node],
        vehicle_states: List[VehicleState],
        dispatch_window_end: float = None
    ) -> Solution:
        """
        Nearest-Neighbor Chain — 從車輛當前位置出發，每次選距離最近的可行請求加到路徑尾端。

        車輛優先順序: MCS-SLOW → MCS-FAST → UAV
        每台車從自身位置出發，不斷挑最近且可行的請求串接，直到沒有可行請求為止。
        """
        c = self.cfg
        solution = Solution(c)

        # 建立 route for each active vehicle
        for vs in vehicle_states:
            if not vs.is_active:
                continue
            route = Route(vehicle_type=vs.vehicle_type, vehicle_id=vs.vehicle_id)
            route.start_position = vs.position
            route.start_time = vs.available_time
            route.start_energy = vs.remaining_energy
            route.is_deployed = len(vs.committed_nodes) > 0

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
                             if n.node_type == 'urgent' and self.compute_uav_delivery(n) > 0 and not n.uav_served]
            before = len(remaining)
            self._nn_fill_route(route, uav_remaining)
            # sync: remove from remaining what UAV took
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
            # 當前位置 = 路徑最後一個 node，或 start_position
            if route.nodes:
                current = route.nodes[-1]
            else:
                current = route.start_position or self.depot

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

    def _try_reserve_activation(
        self,
        solution: Solution,
        unassigned: List[Node],
        vehicle_states: List[VehicleState]
    ) -> List[Node]:
        """
        Reserve MCS Activation (algorithm.md Section 6.3)

        對每個 unassigned 請求, 嘗試從 depot 啟用一台 reserve MCS。
        """
        c = self.cfg

        # 找出所有 reserve vehicles
        reserve_pool = [vs for vs in vehicle_states if not vs.is_active]
        still_unassigned: List[Node] = []

        for node in unassigned:
            activated = False

            for vs in reserve_pool:
                # 建立 candidate route
                candidate_route = Route(vehicle_type=vs.vehicle_type, vehicle_id=vs.vehicle_id)
                candidate_route.start_position = vs.position  # depot
                candidate_route.start_time = vs.available_time
                candidate_route.start_energy = vs.remaining_energy
                candidate_route.insert_node(0, node)

                if self.evaluate_route(candidate_route):
                    # 確認 depot-reachability (evaluate_route 已內含)
                    node.status = 'assigned'
                    activated = True

                    # 加入 solution
                    if vs.vehicle_type == 'uav':
                        solution.add_uav_route(candidate_route)
                    else:
                        solution.add_mcs_route(candidate_route)

                    # 啟用 reserve vehicle
                    vs.is_active = True
                    reserve_pool.remove(vs)
                    print(f"    Reserve activated: {vs.vehicle_type}-{vs.vehicle_id} "
                          f"for request {node.id}")
                    break

            if not activated:
                still_unassigned.append(node)

        return still_unassigned

    def print_config(self) -> None:
        c = self.cfg
        print("="*80)
        print("Experiment Config")
        print("="*80)
        print(f"\nExperimental Variables:")
        print(f"  Arrival process: {'HPP (constant rate)' if c.USE_HPP else 'NHPP (time-varying)'}")
        print(f"  lambda_bar={c.LAMBDA_BAR} req/min ({c.LAMBDA_BAR * 60:.1f} req/hr)")
        print(f"  rho={c.URGENT_RATIO}, DeltaT={c.DELTA_T}min, T={c.T_TOTAL}min")
        if not c.USE_HPP:
            print(f"  r(t) NHPP 24hr profile: {c.NHPP_PROFILE}")
        print(f"\nFleet: S-MCS={c.NUM_MCS_SLOW}, F-MCS={c.NUM_MCS_FAST}, UAV={c.NUM_UAV}")
        print(f"MCS: {c.MCS_SPEED*60:.1f}km/h, {c.MCS_CAPACITY}kWh, "
              f"Slow={c.MCS_SLOW_POWER}kW, Fast={c.MCS_FAST_POWER}kW")
        print(f"UAV: {c.UAV_SPEED*60:.1f}km/h, {c.UAV_DELIVERABLE_ENERGY}kWh, {c.UAV_CHARGE_POWER}kW")
        print("="*80 + "\n")
