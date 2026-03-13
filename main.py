import numpy as np
import random
import time
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Callable, Dict, Optional
from copy import deepcopy


# ==================== 統一實驗配置 ====================
@dataclass
class Config:
    """統一實驗配置 — 所有實驗變數與系統參數集中管理"""

    # === 實驗變數 (Experimental Variables) ===
    RANDOM_SEED: int = 42
    LAMBDA_BAR: float = 0.208          # 平均到達率 (requests/min), 約 12.5 req/hr
    URGENT_RATIO: float = 0.2          # Urgent 需求佔比 ρ
    DELTA_T: int = 10                  # 排程週期 (min)
    T_TOTAL: int = 60                 # 模擬總時長 (min, 12 hr)

    # === NHPP 輪廓 r(t) ===
    PROFILE_OFFPEAK: float = 0.6       # t ∈ [0, 240) 離峰
    PROFILE_PEAK: float = 1.4          # t ∈ [240, 480) 尖峰
    PROFILE_NORMAL: float = 1.0        # t ∈ [480, 720] 平峰

    # === 服務區域 ===
    AREA_SIZE: float = 20.0            # 20 km × 20 km
    DEPOT_X: float = 10.0
    DEPOT_Y: float = 10.0

    # === 需求分布 ===
    DEMAND_MEAN: float = 6.9          # kWh
    DEMAND_STD: float = 4.9           # kWh
    URGENT_ENERGY_MIN: float = 2.0     # kWh
    URGENT_ENERGY_MAX: float = 20.0
    NORMAL_ENERGY_MIN: float = 2.0
    NORMAL_ENERGY_MAX: float = 20.0
    URGENT_TW_MIN: float = 60.0        # min (時間窗寬度)
    URGENT_TW_MAX: float = 90.0
    NORMAL_TW_MIN: float = 480.0       # min
    NORMAL_TW_MAX: float = 600.0

    # === 車隊配置 (初始 active fleet) ===
    NUM_MCS_SLOW: int = 4
    NUM_MCS_FAST: int = 2
    NUM_UAV: int = 0

    # === MCS 共用參數 ===
    MCS_SPEED: float = 0.72            # km/min (43.2 km/h)
    MCS_CAPACITY: float = 100.0        # kWh
    MCS_ENERGY_CONSUMPTION: float = 0.5  # kWh/km
    MCS_ARRIVAL_SETUP_MIN: float = 5.0   # 到達後 setup 時間 (min)
    MCS_CONNECT_MIN: float = 2.0         # 接線時間 (min)
    MCS_DISCONNECT_MIN: float = 2.0      # 拆線時間 (min)

    # === MCS 充電功率 ===
    MCS_SLOW_POWER: float = 11.0       # kW (AC Slow Charging)
    MCS_FAST_POWER: float = 50.0      # kW (DC Super Fast Charging)

    # === UAV 參數 ===
    UAV_SPEED: float = 1.0             # km/min (60 km/h)
    UAV_ENDURANCE: float = 360.0       # min (6 hr)
    UAV_MAX_RANGE: float = 200.0       # km
    UAV_CHARGE_POWER: float = 50.0     # kW
    UAV_DELIVERABLE_ENERGY: float = 20.0  # kWh
    UAV_TAKEOFF_LAND_OVERHEAD: float = 3.0  # min 起降作業開銷

    # === UAV 可服務節點篩選 ===
    UAV_ELIGIBLE_QUANTILE: float = 0.8  # UAV 可服務距離的分位閾值

    # === ALNS 超參數 ===
    ALNS_MAX_ITERATIONS: int = 5000
    ALNS_SEGMENT_SIZE: int = 100
    SA_INITIAL_TEMP: float = 100.0
    SA_COOLING_RATE: float = 0.9995
    SA_FINAL_TEMP: float = 0.01
    SIGMA_1: float = 33.0
    SIGMA_2: float = 13.0
    SIGMA_3: float = 5.0
    DESTROY_RATIO_MIN: float = 0.1
    DESTROY_RATIO_MAX: float = 0.4
    WORST_REMOVAL_P: float = 3.0
    SHAW_REMOVAL_P: float = 6.0
    REACTION_FACTOR: float = 0.1

    # === 目標函數權重 (coverage-first, route-time tie-breaker) ===
    ALPHA_URGENT: float = 50000.0      # Urgent 未服務懲罰
    ALPHA_NORMAL: float = 10000.0      # Normal 未服務懲罰
    ETA_ROUTE_TIME: float = 1.0        # 路徑總時間權重
    GAMMA_DISTANCE: float = 0.01       # 距離權重 (secondary tie-breaker)

    # === Reserve MCS 上限 (0 = 不限) ===
    RESERVE_MCS_SLOW_MAX: int = 0
    RESERVE_MCS_FAST_MAX: int = 0

    def get_arrival_rate_profile(self, t: float) -> float:
        """取得時刻 t 的 nonhomogeneous 輪廓 r(t)"""
        if t < 20.0:
            return self.PROFILE_OFFPEAK
        elif t < 40.0:
            return self.PROFILE_PEAK
        else:
            return self.PROFILE_NORMAL


# ==================== 節點類別 ====================
@dataclass
class Node:
    """節點資料結構"""
    id: int
    x: float
    y: float
    demand: float           # 能量需求 (kWh)
    ready_time: float       # t_i: 請求到達時間
    due_date: float         # d_i: 偏好完成時間 (deadline)
    service_time: float     # 充電時間 (由 demand/power 計算，動態決定)
    node_type: str = 'normal'  # 'depot', 'normal', 'urgent'
    status: str = 'new'        # Request state: new/backlog/assigned/in_service/served/missed


# ==================== 車輛狀態 ====================
@dataclass
class VehicleState:
    """
    車輛 rolling-horizon 狀態

    在每個決策時點，已開始執行的路徑前綴為 frozen-prefix，
    車輛以 release state 表示其最早可重新投入派遣的狀態。
    """
    vehicle_id: int
    vehicle_type: str           # 'mcs_slow', 'mcs_fast', 'uav'
    position: Node              # release_node: 完成當前承諾任務後所在節點
    available_time: float       # release_time: 最早可再接受新任務的時間
    remaining_energy: float     # release_energy: 剩餘能量 (kWh)
    is_active: bool = True      # True=已啟用, False=reserve (depot 待命)
    committed_nodes: List = field(default_factory=list)  # frozen-prefix 中已承諾的節點
    committed_departures: List = field(default_factory=list)  # 每個 committed node 的預計 departure time


# ==================== 路徑容器類別 ====================
class Route:
    """單一路徑容器 (route suffix)"""

    def __init__(self, vehicle_type: str = 'mcs_slow', vehicle_id: int = 0):
        self.vehicle_type: str = vehicle_type
        self.vehicle_id: int = vehicle_id
        self.nodes: List[Node] = []
        self.departure_times: List[float] = []
        self.arrival_times: List[float] = []
        self.user_waiting_times: List[float] = []
        self.mcs_waiting_times: List[float] = []
        self.charging_modes: List[str] = []
        self.total_distance: float = 0.0
        self.total_time: float = 0.0
        self.total_user_waiting_time: float = 0.0
        self.total_mcs_waiting_time: float = 0.0
        self.total_demand: float = 0.0
        self.is_feasible: bool = True

        # Rolling-horizon: route 的起始狀態 (由 VehicleState 提供)
        self.start_position: Optional[Node] = None   # release_node
        self.start_time: float = 0.0                   # release_time
        self.start_energy: float = 0.0                 # release_energy
        self.is_deployed: bool = False                  # 該車輛是否已部署 (有 committed_nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __getitem__(self, index: int) -> Node:
        return self.nodes[index]

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)
        self.total_demand += node.demand

    def insert_node(self, index: int, node: Node) -> None:
        self.nodes.insert(index, node)
        self.total_demand += node.demand

    def remove_node(self, index: int) -> Node:
        node = self.nodes.pop(index)
        self.total_demand -= node.demand
        return node

    def get_node_ids(self) -> List[int]:
        return [node.id for node in self.nodes]

    def copy(self) -> 'Route':
        new_route = Route(self.vehicle_type, self.vehicle_id)
        new_route.nodes = self.nodes.copy()
        new_route.departure_times = self.departure_times.copy()
        new_route.arrival_times = self.arrival_times.copy()
        new_route.user_waiting_times = self.user_waiting_times.copy()
        new_route.mcs_waiting_times = self.mcs_waiting_times.copy()
        new_route.charging_modes = self.charging_modes.copy()
        new_route.total_distance = self.total_distance
        new_route.total_time = self.total_time
        new_route.total_user_waiting_time = self.total_user_waiting_time
        new_route.total_mcs_waiting_time = self.total_mcs_waiting_time
        new_route.total_demand = self.total_demand
        new_route.is_feasible = self.is_feasible
        new_route.start_position = self.start_position
        new_route.start_time = self.start_time
        new_route.start_energy = self.start_energy
        return new_route

    def __repr__(self) -> str:
        node_ids = self.get_node_ids()
        return f"Route({self.vehicle_type}-{self.vehicle_id}): {node_ids}, demand={self.total_demand:.1f}"


# ==================== 解的容器 ====================
class Solution:
    """解的容器 - 包含多條路徑"""

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.mcs_routes: List[Route] = []
        self.uav_routes: List[Route] = []
        self.total_cost: float = float('inf')
        self.total_distance: float = 0.0
        self.total_time: float = 0.0
        self.total_waiting_time: float = 0.0
        self.avg_waiting_time: float = 0.0
        self.avg_waiting_time_urgent: float = 0.0
        self.avg_waiting_time_normal: float = 0.0
        self.coverage_rate: float = 0.0
        self.flexibility_score: float = 0.0
        self.unassigned_nodes: List[Node] = []
        self.total_customers: int = 0
        self.is_feasible: bool = False

    def add_mcs_route(self, route: Route) -> None:
        route.vehicle_id = len(self.mcs_routes)
        self.mcs_routes.append(route)

    def add_uav_route(self, route: Route) -> None:
        route.vehicle_id = len(self.uav_routes)
        self.uav_routes.append(route)

    def get_all_routes(self) -> List[Route]:
        return self.mcs_routes + self.uav_routes

    def get_assigned_node_ids(self) -> Set[int]:
        assigned = set()
        for route in self.get_all_routes():
            assigned.update(route.get_node_ids())
        return assigned

    def calculate_total_cost(self, total_customers: int = None) -> float:
        """
        計算 coverage-first 目標函數值
        Cost = α_u × N_miss_urgent + α_n × N_miss_normal
             + η × T_total + γ × D_total
        """
        all_routes = self.get_all_routes()
        c = self.cfg

        self.total_distance = sum(r.total_distance for r in all_routes)
        self.total_time = sum(r.total_time for r in all_routes)

        urgent_wait_sum, normal_wait_sum = 0.0, 0.0
        urgent_served, normal_served = 0, 0

        for r in all_routes:
            for i, node in enumerate(r.nodes):
                wt = r.user_waiting_times[i] if i < len(r.user_waiting_times) else 0.0
                if node.node_type == 'urgent':
                    urgent_wait_sum += wt
                    urgent_served += 1
                else:
                    normal_wait_sum += wt
                    normal_served += 1

        self.avg_waiting_time_urgent = urgent_wait_sum / urgent_served if urgent_served > 0 else 0.0
        self.avg_waiting_time_normal = normal_wait_sum / normal_served if normal_served > 0 else 0.0

        served_count = urgent_served + normal_served
        self.total_waiting_time = urgent_wait_sum + normal_wait_sum
        self.avg_waiting_time = self.total_waiting_time / served_count if served_count > 0 else 0.0

        if total_customers is not None:
            self.total_customers = total_customers
        if self.total_customers > 0:
            self.coverage_rate = served_count / self.total_customers
        else:
            self.coverage_rate = 1.0

        self.flexibility_score = self._calculate_flexibility_score()

        n_miss_urgent = sum(1 for n in self.unassigned_nodes if n.node_type == 'urgent')
        n_miss_normal = sum(1 for n in self.unassigned_nodes if n.node_type == 'normal')

        unassigned_penalty = c.ALPHA_URGENT * n_miss_urgent + c.ALPHA_NORMAL * n_miss_normal
        route_time_cost = c.ETA_ROUTE_TIME * self.total_time
        distance_cost = c.GAMMA_DISTANCE * self.total_distance

        self.total_cost = unassigned_penalty + route_time_cost + distance_cost
        self.is_feasible = len(self.unassigned_nodes) == 0
        return self.total_cost

    def _calculate_flexibility_score(self) -> float:
        all_routes = self.get_all_routes()
        if len(all_routes) <= 1:
            return 0.0
        loads = [r.total_time for r in all_routes if len(r.nodes) > 0]
        if len(loads) <= 1:
            return 0.0
        mean_load = np.mean(loads)
        if mean_load == 0:
            return 0.0
        return np.std(loads) / mean_load

    def copy(self) -> 'Solution':
        new_solution = Solution(self.cfg)
        new_solution.mcs_routes = [r.copy() for r in self.mcs_routes]
        new_solution.uav_routes = [r.copy() for r in self.uav_routes]
        new_solution.total_cost = self.total_cost
        new_solution.total_distance = self.total_distance
        new_solution.total_time = self.total_time
        new_solution.total_waiting_time = self.total_waiting_time
        new_solution.avg_waiting_time = self.avg_waiting_time
        new_solution.avg_waiting_time_urgent = self.avg_waiting_time_urgent
        new_solution.avg_waiting_time_normal = self.avg_waiting_time_normal
        new_solution.coverage_rate = self.coverage_rate
        new_solution.flexibility_score = self.flexibility_score
        new_solution.total_customers = self.total_customers
        new_solution.unassigned_nodes = self.unassigned_nodes.copy()
        new_solution.is_feasible = self.is_feasible
        return new_solution

    def __repr__(self) -> str:
        return (f"Solution(MCS={len(self.mcs_routes)}, UAV={len(self.uav_routes)}, "
                f"cost={self.total_cost:.2f}, coverage={self.coverage_rate:.1%})")

    def print_summary(self) -> None:
        c = self.cfg
        print("\n" + "="*60)
        print("Solution Summary")
        print("="*60)

        n_miss_urgent = sum(1 for n in self.unassigned_nodes if n.node_type == 'urgent')
        n_miss_normal = sum(1 for n in self.unassigned_nodes if n.node_type == 'normal')

        print(f"\nObjective (Coverage-First)")
        print(f"  Total Cost: {self.total_cost:.2f}")
        print(f"  Urgent miss: {n_miss_urgent}, Normal miss: {n_miss_normal}")

        served = self.total_customers - len(self.unassigned_nodes)
        print(f"\nKey Metrics")
        print(f"  Coverage: {self.coverage_rate:.1%} ({served}/{self.total_customers})")
        print(f"  Total Route Time: {self.total_time:.2f} min")
        print(f"  Avg Wait (All): {self.avg_waiting_time:.2f} min")
        print(f"  Avg Wait (Urgent): {self.avg_waiting_time_urgent:.2f} min")
        print(f"  Avg Wait (Normal): {self.avg_waiting_time_normal:.2f} min")
        print(f"  Total Distance: {self.total_distance:.1f} km")

        for route in self.mcs_routes:
            if len(route.nodes) == 0:
                continue
            mode_summary = f"[F:{route.charging_modes.count('FAST')}/S:{route.charging_modes.count('SLOW')}]"
            print(f"  {route.vehicle_type.upper()}-{route.vehicle_id}: "
                  f"nodes={route.get_node_ids()}, dist={route.total_distance:.1f}km, "
                  f"wait={route.total_user_waiting_time:.1f}min {mode_summary}")
        for route in self.uav_routes:
            if len(route.nodes) == 0:
                continue
            print(f"  UAV-{route.vehicle_id}: nodes={route.get_node_ids()}, "
                  f"dist={route.total_distance:.1f}km, wait={route.total_user_waiting_time:.1f}min")
        print("="*60)


# ==================== 距離矩陣快取 ====================
class DistanceMatrix:
    """預計算距離矩陣 (MCS: 曼哈頓, UAV: 歐氏)"""

    def __init__(self, nodes: List[Node]):
        n = len(nodes)
        self.n = n
        self.nodes = nodes
        self.id_to_idx = {node.id: idx for idx, node in enumerate(nodes)}
        self.manhattan = np.zeros((n, n))
        self.euclidean = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = abs(nodes[i].x - nodes[j].x)
                    dy = abs(nodes[i].y - nodes[j].y)
                    self.manhattan[i, j] = dx + dy
                    self.euclidean[i, j] = np.sqrt(dx**2 + dy**2)

    def get_distance(self, node1: Node, node2: Node, vehicle: str = 'mcs_slow') -> float:
        idx1 = self.id_to_idx.get(node1.id)
        idx2 = self.id_to_idx.get(node2.id)
        if idx1 is None or idx2 is None:
            # fallback for nodes not in matrix
            dx = abs(node1.x - node2.x)
            dy = abs(node1.y - node2.y)
            if vehicle in ('mcs_fast', 'mcs_slow'):
                return dx + dy
            else:
                return np.sqrt(dx**2 + dy**2)
        if vehicle in ('mcs_fast', 'mcs_slow'):
            return self.manhattan[idx1, idx2]
        else:
            return self.euclidean[idx1, idx2]


# ==================== 主程式類別 ====================
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

    # ==================== NHPP 需求生成 ====================
    def generate_requests(self, slot_start: float) -> List[Node]:
        """依 NHPP 生成單一排程週期的充電請求"""
        c = self.cfg
        r_t = c.get_arrival_rate_profile(slot_start)
        mu_k = c.LAMBDA_BAR * r_t * c.DELTA_T
        n_requests = np.random.poisson(mu_k)

        new_nodes: List[Node] = []
        for _ in range(n_requests):
            node_id = self._next_node_id
            self._next_node_id += 1

            x = np.random.uniform(0, c.AREA_SIZE)
            y = np.random.uniform(0, c.AREA_SIZE)
            t_i = np.random.uniform(slot_start, slot_start + c.DELTA_T)

            is_urgent = np.random.random() < c.URGENT_RATIO
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

            node = Node(
                id=node_id, x=x, y=y, demand=demand,
                ready_time=t_i, due_date=d_i, service_time=0,
                node_type=node_type, status='new'
            )
            new_nodes.append(node)
        return new_nodes

    def setup_nodes(self, customer_nodes: List[Node]) -> None:
        """設定節點列表並建立距離矩陣"""
        self.nodes = [self.depot] + customer_nodes
        self.dist_matrix = DistanceMatrix(self.nodes)
        self._compute_uav_eligible()

    def _compute_uav_eligible(self) -> None:
        c = self.cfg
        customers = [n for n in self.nodes if n.node_type != 'depot']
        if not customers:
            return
        dists = []
        for cust in customers:
            d = np.sqrt((cust.x - self.depot.x)**2 + (cust.y - self.depot.y)**2)
            dists.append((cust.id, d))
        dist_values = [d for _, d in dists]
        threshold = np.quantile(dist_values, c.UAV_ELIGIBLE_QUANTILE) if dist_values else 0
        self.uav_eligible_ids = {cid for cid, d in dists if d >= threshold}

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

    def get_charging_profile(self, vehicle: str, node_type: str, demand: float) -> Tuple[str, float]:
        c = self.cfg
        if vehicle == 'uav':
            return 'FAST', self.calculate_charging_time(demand, c.UAV_CHARGE_POWER)
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

        if route.total_demand > capacity:
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

            mode, charging_time = self.get_charging_profile(vehicle, node.node_type, node.demand)
            departure_time = service_start + charging_time + disconnect

            if departure_time > node.due_date:
                route.is_feasible = False
                return False

            user_waiting_time = max(0.0, arrival_time - node.ready_time)
            vehicle_idle_time = max(0.0, node.ready_time - earliest_service)

            # 能量扣減 (MCS only)
            if self._is_mcs(vehicle):
                travel_energy = distance * c.MCS_ENERGY_CONSUMPTION
                current_energy -= (travel_energy + node.demand)

            route.arrival_times.append(arrival_time)
            route.departure_times.append(departure_time)
            route.charging_modes.append(mode)
            route.user_waiting_times.append(user_waiting_time)
            route.mcs_waiting_times.append(vehicle_idle_time)
            route.total_user_waiting_time += user_waiting_time

            current_time = departure_time
            prev_node = node

        # depot-reachability check: 完成 suffix 後是否能返回 depot
        if self._is_mcs(vehicle):
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
        if route.total_demand + node.demand > capacity:
            return False, float('inf')

        if vehicle == 'uav' and node.id not in self.uav_eligible_ids:
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

    # ==================== 建構啟發式 (無分群, 使用 vehicle release state) ====================
    def greedy_insertion_construction(
        self,
        active_requests: List[Node],
        vehicle_states: List[VehicleState],
        dispatch_window_end: float = None
    ) -> Solution:
        """
        Greedy Insertion Construction — Bin-Packing Priority

        優先塞滿已部署車輛 (有 committed_nodes 或本輪已分配節點)，
        只有在所有已部署車輛都硬約束不可行時，才開新車。

        Cascade:
          Step 1: 已部署 SLOW → Step 2: 已部署 FAST →
          Step 3: 未部署 SLOW (開新車) → Step 4: 未部署 FAST →
          Step 5: UAV → Phase 3: Reserve activation
        """
        c = self.cfg
        solution = Solution(c)

        # 為每台 active vehicle 建立 route (suffix)
        vehicle_route_map: Dict[int, Tuple[str, int]] = {}  # vehicle_id -> ('mcs'|'uav', route_idx)

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

        # Phase 1: 排序 — 按 due_date, 再按 ready_time
        sorted_requests = sorted(active_requests, key=lambda n: (n.due_date, n.ready_time))
        unassigned: List[Node] = []

        # Phase 2: 逐一插入 (優先 SLOW, 不可行才 fallback FAST)
        for node in sorted_requests:
            best_route_type: Optional[str] = None
            best_route_idx = -1
            best_position = -1
            min_marginal_cost = float('inf')

            # === Bin-Packing Priority Cascade ===
            # 已部署 = committed_nodes 非空 或本輪已有分配節點
            # 優先塞滿已部署車輛，只有硬約束不可行才開新車

            # Step 1: 已部署 SLOW
            for r_idx, route in enumerate(solution.mcs_routes):
                if route.vehicle_type != 'mcs_slow':
                    continue
                if not (route.is_deployed or len(route.nodes) > 0):
                    continue
                for pos in range(len(route.nodes) + 1):
                    feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
                    if feasible and delta_cost < min_marginal_cost:
                        min_marginal_cost = delta_cost
                        best_route_type = 'mcs'
                        best_route_idx = r_idx
                        best_position = pos

            # Step 2: 已部署 FAST（Step 1 全部不可行才到這）
            if best_route_type is None:
                for r_idx, route in enumerate(solution.mcs_routes):
                    if route.vehicle_type != 'mcs_fast':
                        continue
                    if not (route.is_deployed or len(route.nodes) > 0):
                        continue
                    for pos in range(len(route.nodes) + 1):
                        feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
                        if feasible and delta_cost < min_marginal_cost:
                            min_marginal_cost = delta_cost
                            best_route_type = 'mcs'
                            best_route_idx = r_idx
                            best_position = pos

            # Step 3: 未部署 SLOW — 開新車（Step 1+2 都不可行才到這）
            if best_route_type is None:
                for r_idx, route in enumerate(solution.mcs_routes):
                    if route.vehicle_type != 'mcs_slow':
                        continue
                    if route.is_deployed or len(route.nodes) > 0:
                        continue  # 已在 Step 1 嘗試過
                    for pos in range(len(route.nodes) + 1):
                        feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
                        if feasible and delta_cost < min_marginal_cost:
                            min_marginal_cost = delta_cost
                            best_route_type = 'mcs'
                            best_route_idx = r_idx
                            best_position = pos

            # Step 4: 未部署 FAST — 開新車
            if best_route_type is None:
                for r_idx, route in enumerate(solution.mcs_routes):
                    if route.vehicle_type != 'mcs_fast':
                        continue
                    if route.is_deployed or len(route.nodes) > 0:
                        continue  # 已在 Step 2 嘗試過
                    for pos in range(len(route.nodes) + 1):
                        feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
                        if feasible and delta_cost < min_marginal_cost:
                            min_marginal_cost = delta_cost
                            best_route_type = 'mcs'
                            best_route_idx = r_idx
                            best_position = pos

            # Step 5: 嘗試 UAV 路徑 (僅 urgent + uav_eligible)
            if node.node_type == 'urgent' and node.id in self.uav_eligible_ids:
                for r_idx, route in enumerate(solution.uav_routes):
                    for pos in range(len(route.nodes) + 1):
                        feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
                        if feasible and delta_cost < min_marginal_cost:
                            min_marginal_cost = delta_cost
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

        # Phase 3: Reserve MCS Activation — Section 6.3
        if unassigned:
            unassigned = self._try_reserve_activation(
                solution, unassigned, vehicle_states
            )

        solution.unassigned_nodes = unassigned
        solution.calculate_total_cost(len(active_requests))
        return solution

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
        print(f"  lambda_bar={c.LAMBDA_BAR} req/min ({c.LAMBDA_BAR * 60:.1f} req/hr)")
        print(f"  rho={c.URGENT_RATIO}, DeltaT={c.DELTA_T}min, T={c.T_TOTAL}min")
        print(f"  r(t): offpeak={c.PROFILE_OFFPEAK}, peak={c.PROFILE_PEAK}, normal={c.PROFILE_NORMAL}")
        print(f"\nFleet: S-MCS={c.NUM_MCS_SLOW}, F-MCS={c.NUM_MCS_FAST}, UAV={c.NUM_UAV}")
        print(f"MCS: {c.MCS_SPEED*60:.1f}km/h, {c.MCS_CAPACITY}kWh, "
              f"Slow={c.MCS_SLOW_POWER}kW, Fast={c.MCS_FAST_POWER}kW")
        print(f"UAV: {c.UAV_SPEED*60:.1f}km/h, {c.UAV_DELIVERABLE_ENERGY}kWh, {c.UAV_CHARGE_POWER}kW")
        print("="*80 + "\n")


# ==================== ALNS 求解器 ====================
class ALNSSolver:
    """ALNS 求解器 (保留, 可用於 slot 內改善)"""

    def __init__(self, problem: ChargingSchedulingProblem, cfg: Config = None):
        self.problem = problem
        self.cfg = cfg or problem.cfg

        self.destroy_ops: Dict[str, Callable] = {
            'random_removal': self._random_removal,
            'worst_removal':  self._worst_removal,
        }
        self.repair_ops: Dict[str, Callable] = {
            'greedy_insertion': self._greedy_insertion,
            'regret2_insertion': self._regret2_insertion,
        }
        self.destroy_weights: Dict[str, float] = {k: 1.0 for k in self.destroy_ops}
        self.repair_weights:  Dict[str, float] = {k: 1.0 for k in self.repair_ops}
        self._reset_segment_stats()

    def solve(self, initial_solution: Solution) -> Solution:
        c = self.cfg
        current = initial_solution.copy()
        best = initial_solution.copy()
        temperature = c.SA_INITIAL_TEMP
        total_customers = current.total_customers

        served_count = sum(len(r.nodes) for r in current.get_all_routes())
        min_remove = max(1, int(served_count * c.DESTROY_RATIO_MIN))
        max_remove = max(min_remove + 1, int(served_count * c.DESTROY_RATIO_MAX))

        for iteration in range(1, c.ALNS_MAX_ITERATIONS + 1):
            d_name = self._roulette_select(self.destroy_weights)
            r_name = self._roulette_select(self.repair_weights)

            candidate = current.copy()
            num_remove = random.randint(min_remove, max_remove)
            removed = self.destroy_ops[d_name](candidate, num_remove)
            self.repair_ops[r_name](candidate, removed)
            candidate.calculate_total_cost(total_customers)

            delta = candidate.total_cost - current.total_cost
            if delta < -1e-9:
                current = candidate
                if current.total_cost < best.total_cost - 1e-9:
                    best = current.copy()
                    self._record_score(d_name, r_name, c.SIGMA_1)
                else:
                    self._record_score(d_name, r_name, c.SIGMA_2)
            elif self._sa_accept(delta, temperature):
                current = candidate
                self._record_score(d_name, r_name, c.SIGMA_3)

            temperature = max(temperature * c.SA_COOLING_RATE, c.SA_FINAL_TEMP)
            if iteration % c.ALNS_SEGMENT_SIZE == 0:
                self._update_weights()
                self._reset_segment_stats()

        return best

    def _roulette_select(self, weights: Dict[str, float]) -> str:
        names = list(weights.keys())
        vals = [weights[n] for n in names]
        total = sum(vals)
        r = random.random() * total
        cumulative = 0.0
        for name, val in zip(names, vals):
            cumulative += val
            if r <= cumulative:
                return name
        return names[-1]

    def _sa_accept(self, delta: float, temperature: float) -> bool:
        if temperature < 1e-12:
            return False
        return random.random() < np.exp(-delta / temperature)

    def _reset_segment_stats(self):
        self._destroy_scores = {k: 0.0 for k in self.destroy_ops}
        self._repair_scores  = {k: 0.0 for k in self.repair_ops}
        self._destroy_usage  = {k: 0   for k in self.destroy_ops}
        self._repair_usage   = {k: 0   for k in self.repair_ops}

    def _record_score(self, d_name, r_name, score):
        self._destroy_scores[d_name] += score
        self._repair_scores[r_name]  += score
        self._destroy_usage[d_name]  += 1
        self._repair_usage[r_name]   += 1

    def _update_weights(self):
        lam = self.cfg.REACTION_FACTOR
        for name in self.destroy_weights:
            usage = self._destroy_usage[name]
            if usage > 0:
                avg = self._destroy_scores[name] / usage
                self.destroy_weights[name] = (1 - lam) * self.destroy_weights[name] + lam * avg
            self.destroy_weights[name] = max(self.destroy_weights[name], 0.1)
        for name in self.repair_weights:
            usage = self._repair_usage[name]
            if usage > 0:
                avg = self._repair_scores[name] / usage
                self.repair_weights[name] = (1 - lam) * self.repair_weights[name] + lam * avg
            self.repair_weights[name] = max(self.repair_weights[name], 0.1)

    # --- Destroy ---
    def _random_removal(self, solution: Solution, num_remove: int) -> List[Node]:
        candidates = []
        for r_idx, r in enumerate(solution.mcs_routes):
            for n_pos in range(len(r.nodes)):
                candidates.append(('mcs', r_idx, n_pos))
        for r_idx, r in enumerate(solution.uav_routes):
            for n_pos in range(len(r.nodes)):
                candidates.append(('uav', r_idx, n_pos))
        if not candidates:
            return []
        num_remove = min(num_remove, len(candidates))
        selected = random.sample(candidates, num_remove)
        selected.sort(key=lambda x: x[2], reverse=True)
        removed = []
        for v_type, r_idx, n_pos in selected:
            routes = solution.mcs_routes if v_type == 'mcs' else solution.uav_routes
            removed.append(routes[r_idx].remove_node(n_pos))
        for r in solution.get_all_routes():
            self.problem.evaluate_route(r)
        return removed

    def _worst_removal(self, solution: Solution, num_remove: int) -> List[Node]:
        node_costs = []
        for v_type, routes in [('mcs', solution.mcs_routes), ('uav', solution.uav_routes)]:
            for r_idx, r in enumerate(routes):
                for n_pos, node in enumerate(r.nodes):
                    cost = r.user_waiting_times[n_pos] if n_pos < len(r.user_waiting_times) else 0.0
                    node_costs.append((cost, v_type, r_idx, n_pos))
        if not node_costs:
            return []
        node_costs.sort(key=lambda x: x[0], reverse=True)
        to_remove = []
        num_remove = min(num_remove, len(node_costs))
        for _ in range(num_remove):
            if not node_costs:
                break
            idx = int(random.random() ** self.cfg.WORST_REMOVAL_P * len(node_costs))
            idx = min(idx, len(node_costs) - 1)
            to_remove.append(node_costs.pop(idx))
        to_remove.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        removed = []
        for cost, v_type, r_idx, n_pos in to_remove:
            routes = solution.mcs_routes if v_type == 'mcs' else solution.uav_routes
            removed.append(routes[r_idx].remove_node(n_pos))
        for r in solution.get_all_routes():
            self.problem.evaluate_route(r)
        return removed

    # --- Repair ---
    def _greedy_insertion(self, solution: Solution, removed_nodes: List[Node]):
        pool = removed_nodes + solution.unassigned_nodes
        pool.sort(key=lambda n: (0 if n.node_type == 'urgent' else 1, n.due_date))
        still_unassigned = []
        for node in pool:
            best_type, best_idx, best_pos, min_cost = None, -1, -1, float('inf')
            # 優先 SLOW
            for r_idx, r in enumerate(solution.mcs_routes):
                if r.vehicle_type != 'mcs_slow':
                    continue
                for pos in range(len(r.nodes) + 1):
                    ok, dc = self.problem.incremental_insertion_check(r, pos, node)
                    if ok and dc < min_cost:
                        min_cost, best_type, best_idx, best_pos = dc, 'mcs', r_idx, pos
            # SLOW 不可行 → fallback FAST
            if best_type is None:
                for r_idx, r in enumerate(solution.mcs_routes):
                    if r.vehicle_type != 'mcs_fast':
                        continue
                    for pos in range(len(r.nodes) + 1):
                        ok, dc = self.problem.incremental_insertion_check(r, pos, node)
                        if ok and dc < min_cost:
                            min_cost, best_type, best_idx, best_pos = dc, 'mcs', r_idx, pos
            if node.node_type == 'urgent':
                for r_idx, r in enumerate(solution.uav_routes):
                    for pos in range(len(r.nodes) + 1):
                        ok, dc = self.problem.incremental_insertion_check(r, pos, node)
                        if ok and dc < min_cost:
                            min_cost, best_type, best_idx, best_pos = dc, 'uav', r_idx, pos
            if best_type:
                routes = solution.mcs_routes if best_type == 'mcs' else solution.uav_routes
                routes[best_idx].insert_node(best_pos, node)
                if not self.problem.evaluate_route(routes[best_idx]):
                    routes[best_idx].remove_node(best_pos)
                    self.problem.evaluate_route(routes[best_idx])
                    still_unassigned.append(node)
            else:
                still_unassigned.append(node)
        solution.unassigned_nodes = still_unassigned

    def _regret2_insertion(self, solution: Solution, removed_nodes: List[Node]):
        pool = removed_nodes + solution.unassigned_nodes
        still_unassigned = []
        while pool:
            best_node, best_n_idx = None, -1
            best_regret, best_type, best_idx, best_pos = -float('inf'), None, -1, -1
            best_cost = float('inf')
            for n_idx, node in enumerate(pool):
                # 優先 SLOW, 不可行才 fallback FAST
                slow_options = []
                for r_idx, r in enumerate(solution.mcs_routes):
                    if r.vehicle_type != 'mcs_slow':
                        continue
                    for pos in range(len(r.nodes) + 1):
                        ok, dc = self.problem.incremental_insertion_check(r, pos, node)
                        if ok:
                            slow_options.append((dc, 'mcs', r_idx, pos))
                if slow_options:
                    options = slow_options
                else:
                    options = []
                    for r_idx, r in enumerate(solution.mcs_routes):
                        if r.vehicle_type != 'mcs_fast':
                            continue
                        for pos in range(len(r.nodes) + 1):
                            ok, dc = self.problem.incremental_insertion_check(r, pos, node)
                            if ok:
                                options.append((dc, 'mcs', r_idx, pos))
                if node.node_type == 'urgent':
                    for r_idx, r in enumerate(solution.uav_routes):
                        for pos in range(len(r.nodes) + 1):
                            ok, dc = self.problem.incremental_insertion_check(r, pos, node)
                            if ok:
                                options.append((dc, 'uav', r_idx, pos))
                if not options:
                    continue
                options.sort(key=lambda x: x[0])
                c1, t1, r1, p1 = options[0]
                c2 = options[1][0] if len(options) >= 2 else c1 + 1e6
                regret = c2 - c1
                if regret > best_regret + 1e-9 or (abs(regret - best_regret) < 1e-9 and c1 < best_cost - 1e-9):
                    best_regret, best_node, best_n_idx = regret, node, n_idx
                    best_type, best_idx, best_pos, best_cost = t1, r1, p1, c1
            if best_node is None:
                still_unassigned.extend(pool)
                break
            routes = solution.mcs_routes if best_type == 'mcs' else solution.uav_routes
            routes[best_idx].insert_node(best_pos, best_node)
            if not self.problem.evaluate_route(routes[best_idx]):
                routes[best_idx].remove_node(best_pos)
                self.problem.evaluate_route(routes[best_idx])
                still_unassigned.append(best_node)
            pool.pop(best_n_idx)
        solution.unassigned_nodes = still_unassigned


# ==================== 批次調度模擬 ====================
def initialize_fleet(cfg: Config, depot: Node) -> List[VehicleState]:
    """初始化車隊 — 所有車輛在 t=0 位於 depot 待命"""
    fleet: List[VehicleState] = []
    vid = 0

    for i in range(cfg.NUM_MCS_SLOW):
        fleet.append(VehicleState(
            vehicle_id=vid, vehicle_type='mcs_slow',
            position=depot, available_time=0.0,
            remaining_energy=cfg.MCS_CAPACITY, is_active=True
        ))
        vid += 1

    for i in range(cfg.NUM_MCS_FAST):
        fleet.append(VehicleState(
            vehicle_id=vid, vehicle_type='mcs_fast',
            position=depot, available_time=0.0,
            remaining_energy=cfg.MCS_CAPACITY, is_active=True
        ))
        vid += 1

    for i in range(cfg.NUM_UAV):
        fleet.append(VehicleState(
            vehicle_id=vid, vehicle_type='uav',
            position=depot, available_time=0.0,
            remaining_energy=cfg.UAV_DELIVERABLE_ENERGY, is_active=True
        ))
        vid += 1

    # Reserve pool (unlimited by default)
    for i in range(10):  # 10 reserve slow-MCS
        fleet.append(VehicleState(
            vehicle_id=vid, vehicle_type='mcs_slow',
            position=depot, available_time=0.0,
            remaining_energy=cfg.MCS_CAPACITY, is_active=False
        ))
        vid += 1

    for i in range(5):   # 5 reserve fast-MCS
        fleet.append(VehicleState(
            vehicle_id=vid, vehicle_type='mcs_fast',
            position=depot, available_time=0.0,
            remaining_energy=cfg.MCS_CAPACITY, is_active=False
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
        if not vs.is_active:
            continue

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
            _, charging_time = problem.get_charging_profile(vehicle, node.node_type, node.demand)
            departure = service_start + charging_time + disconnect

            if departure <= slot_end_time:
                # 完成服務
                node.status = 'served'
                if problem._is_mcs(vehicle):
                    current_energy -= (dist * c.MCS_ENERGY_CONSUMPTION + node.demand)
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


# ==================== 主程式入口 ====================
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
        if not vs.is_active:
            continue

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
            _, charging_time = problem.get_charging_profile(vehicle, node.node_type, node.demand)
            departure = service_start + charging_time + disconnect

            if departure <= execution_end_time:
                node.status = 'served'
                if problem._is_mcs(vehicle):
                    current_energy -= (dist * c.MCS_ENERGY_CONSUMPTION + node.demand)
                current_time = departure
                current_pos = node
            elif arrival < execution_end_time:
                node.status = 'in_service'
                new_committed.append(node)
                new_committed_departures.append(departure)
                if problem._is_mcs(vehicle):
                    current_energy -= (dist * c.MCS_ENERGY_CONSUMPTION + node.demand)
                current_time = departure
                current_pos = node
            else:
                # arrival >= execution_end_time
                if current_time < execution_end_time:
                    # MCS 已出發但尚未抵達 → en-route, 加入 committed
                    node.status = 'in_service'
                    new_committed.append(node)
                    new_committed_departures.append(departure)
                    if problem._is_mcs(vehicle):
                        current_energy -= (dist * c.MCS_ENERGY_CONSUMPTION + node.demand)
                    current_time = departure
                    current_pos = node
                break

        vs.position = current_pos
        vs.available_time = current_time
        vs.remaining_energy = current_energy
        vs.committed_nodes = new_committed
        vs.committed_departures = new_committed_departures


def main():
    cfg = Config()
    problem = ChargingSchedulingProblem(cfg)
    problem.print_config()

    # 初始化車隊
    fleet = initialize_fleet(cfg, problem.depot)
    active_count = sum(1 for vs in fleet if vs.is_active)
    reserve_count = sum(1 for vs in fleet if not vs.is_active)
    print(f"Fleet initialized: active={active_count}, reserve={reserve_count}")

    # 請求池
    all_requests: List[Node] = []  # 所有已生成的請求 (歷史紀錄)
    backlog: List[Node] = []       # 上一 slot 未服務的 backlog

    # 統計
    total_generated = 0
    total_served = 0
    total_missed = 0
    slot_logs: List[Dict] = []

    num_slots = cfg.T_TOTAL // cfg.DELTA_T

    print(f"\n{'='*80}")
    print(f"Batch Dispatching | {num_slots} slots x {cfg.DELTA_T}min = {cfg.T_TOTAL}min")
    print(f"{'='*80}\n")

    for k in range(1, num_slots + 1):
        slot_start = (k - 1) * cfg.DELTA_T
        slot_end = k * cfg.DELTA_T
        dispatch_time = slot_end
        next_decision_time = slot_end + cfg.DELTA_T

        # Step 1: 生成本期新請求
        new_requests = problem.generate_requests(slot_start)
        total_generated += len(new_requests)

        # Step 2: 處理 backlog — 逾期者標記 missed
        active_backlog: List[Node] = []
        slot_missed = 0
        for req in backlog:
            if req.due_date <= dispatch_time:
                req.status = 'missed'
                total_missed += 1
                slot_missed += 1
            else:
                req.status = 'backlog'
                active_backlog.append(req)

        # Active request pool = new + active_backlog
        active_pool = new_requests + active_backlog

        # Step 3a: 更新 committed nodes (runs every slot, even if pool is empty)
        completed_committed = 0
        for vs in fleet:
            if vs.is_active:
                if vs.committed_nodes:
                    still_committed = []
                    still_departures = []
                    for node, dep_time in zip(vs.committed_nodes, vs.committed_departures):
                        if dep_time <= dispatch_time:
                            node.status = 'served'
                            completed_committed += 1
                        else:
                            still_committed.append(node)
                            still_departures.append(dep_time)
                    vs.committed_nodes = still_committed
                    vs.committed_departures = still_departures
                vs.available_time = max(vs.available_time, dispatch_time)
            else:
                vs.available_time = slot_end  # reserve 待命
        total_served += completed_committed

        if not active_pool:
            r_t = cfg.get_arrival_rate_profile(slot_start)
            active_vehicles = sum(1 for vs in fleet if vs.is_active)
            print(f"  Slot {k:3d} [t={slot_start:.0f}-{slot_end:.0f}] r(t)={r_t:.1f} | "
                  f"new=0, backlog=0, pool=0 | "
                  f"served={completed_committed}, missed={slot_missed} | "
                  f"fleet={active_vehicles}")
            slot_logs.append({
                'slot': k, 'new': 0, 'backlog': 0, 'pool': 0,
                'assigned': 0, 'served': completed_committed,
                'missed': slot_missed, 'backlog_out': 0,
                'active_fleet': active_vehicles
            })
            continue

        # Step 3b: 更新距離矩陣 (加入新節點)
        all_customer_nodes = [n for n in all_requests if n.status not in ('served', 'missed')]
        all_customer_nodes.extend(new_requests)
        all_requests.extend(new_requests)
        problem.setup_nodes(all_customer_nodes)

        # Step 4: 計算 vehicle release states

        # Step 5: Greedy insertion + reserve activation
        solution = problem.greedy_insertion_construction(active_pool, fleet, dispatch_window_end=next_decision_time)

        n_assigned = sum(1 for n in active_pool if n.status == 'assigned')
        n_unassigned = len(solution.unassigned_nodes)

        # Step 6: 模擬執行
        simulate_dispatch_window(solution, fleet, problem, next_decision_time)

        # Step 7: 更新 backlog
        new_backlog: List[Node] = []
        for node in solution.unassigned_nodes:
            if node.due_date > next_decision_time:
                node.status = 'backlog'
                new_backlog.append(node)
            else:
                node.status = 'missed'
                total_missed += 1
                slot_missed += 1

        # 也檢查 assigned 但未完成服務的
        for route in solution.get_all_routes():
            for node in route.nodes:
                if node.status == 'assigned':
                    # 尚未開始服務 → backlog
                    node.status = 'backlog'
                    new_backlog.append(node)

        backlog = new_backlog
        # 計算本 slot 完成的服務: simulate 中直接 served 的
        # (committed 完成已在 Step 3a 計入 total_served)
        direct_served = 0
        for route in solution.get_all_routes():
            for node in route.nodes:
                if node.status == 'served':
                    direct_served += 1
        slot_served = completed_committed + direct_served
        total_served += direct_served

        r_t = cfg.get_arrival_rate_profile(slot_start)
        active_vehicles = sum(1 for vs in fleet if vs.is_active)
        print(f"  Slot {k:3d} [t={slot_start:.0f}-{slot_end:.0f}] r(t)={r_t:.1f} | "
              f"new={len(new_requests)}, backlog_in={len(active_backlog)}, "
              f"pool={len(active_pool)} | "
              f"assigned={n_assigned}, served={slot_served}, "
              f"missed={slot_missed}, backlog_out={len(new_backlog)} | "
              f"fleet={active_vehicles}")

        slot_logs.append({
            'slot': k, 'new': len(new_requests),
            'backlog_in': len(active_backlog), 'pool': len(active_pool),
            'assigned': n_assigned, 'served': slot_served,
            'missed': slot_missed, 'backlog_out': len(new_backlog),
            'active_fleet': active_vehicles
        })

    # ==================== Terminal: sweep remaining committed nodes ====================
    # Nodes still in committed_nodes are in_service — they will complete; count as served.
    terminal_committed = 0
    for vs in fleet:
        for node in vs.committed_nodes:
            node.status = 'served'
            terminal_committed += 1
        vs.committed_nodes = []
        vs.committed_departures = []
    total_served += terminal_committed

    # ==================== 最終統計 ====================
    accounted = total_served + total_missed + len(backlog)
    print(f"\n{'='*80}")
    print("Simulation Complete -- Summary")
    print(f"{'='*80}")
    print(f"  Total Generated: {total_generated}")
    print(f"  Total Served: {total_served}")
    print(f"  Total Missed: {total_missed}")
    print(f"  Remaining Backlog: {len(backlog)}")
    if terminal_committed > 0:
        print(f"  (includes {terminal_committed} committed/in-service at terminal time)")
    if accounted != total_generated:
        print(f"  WARNING: accounting mismatch — accounted={accounted}, generated={total_generated}")
    if total_generated > 0:
        print(f"  Coverage: {total_served / total_generated:.1%}")
        print(f"  Miss Ratio: {total_missed / total_generated:.1%}")

    active_vehicles = sum(1 for vs in fleet if vs.is_active)
    print(f"  Final Active Vehicles: {active_vehicles}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
