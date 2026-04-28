import numpy as np
from dataclasses import dataclass
from typing import List, Set

from config import Config


# ==================== 節點類別 ====================
@dataclass
class Node:
    """節點資料結構 (static single-batch)"""
    id: int
    x: float
    y: float
    demand: float           # 能量需求 (kWh)
    ready_time: float       # t_i: 請求到達時間
    due_date: float         # d_i: 偏好完成時間 (deadline)
    service_time: float     # 充電時間 (由 demand/power 計算，動態決定)
    node_type: str = 'normal'  # 'depot', 'normal', 'urgent'
    status: str = 'new'        # 'new' | 'assigned' | 'served'
    ev_current_soc: float = 0.0        # EV 當前 SOC (0.0-1.0)


# ==================== 車輛狀態 ====================
@dataclass
class VehicleState:
    """車輛狀態 (static single-batch — 全部於 t=0 在 depot 待命)"""
    vehicle_id: int
    vehicle_type: str           # 'mcs_slow', 'mcs_fast', 'uav'
    position: Node              # 起始節點 (depot)
    available_time: float       # 起始時刻 (固定為 0)
    remaining_energy: float     # 起始能量 (kWh)


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
        self.mcs_routes.append(route)

    def add_uav_route(self, route: Route) -> None:
        self.uav_routes.append(route)

    def get_all_routes(self) -> List[Route]:
        return self.mcs_routes + self.uav_routes

    def get_assigned_node_ids(self) -> Set[int]:
        assigned = set()
        for route in self.get_all_routes():
            assigned.update(route.get_node_ids())
        return assigned

    def get_all_assigned_nodes_with_positions(self) -> List[tuple]:
        """Return [(node, route_type, route_idx, node_pos), ...] for all assigned nodes."""
        result = []
        for r_idx, r in enumerate(self.mcs_routes):
            for n_pos, node in enumerate(r.nodes):
                result.append((node, 'mcs', r_idx, n_pos))
        for r_idx, r in enumerate(self.uav_routes):
            for n_pos, node in enumerate(r.nodes):
                result.append((node, 'uav', r_idx, n_pos))
        return result

    def calculate_total_cost(self, total_customers: int = None) -> float:
        """
        目標函數 — 三項可解釋成本

        Z = α_u · N_miss_urgent  +  α_n · N_miss_normal     (未服務懲罰)
          + β_u · W_urgent_total +  β_n · W_normal_total     (回應時間成本)
          + γ   · D_total                                    (行駛成本)
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

        n_miss_urgent = sum(1 for n in self.unassigned_nodes if n.node_type == 'urgent')
        n_miss_normal = sum(1 for n in self.unassigned_nodes if n.node_type == 'normal')

        miss_penalty = c.ALPHA_URGENT * n_miss_urgent + c.ALPHA_NORMAL * n_miss_normal
        response_cost = (c.BETA_WAITING_URGENT * urgent_wait_sum
                         + c.BETA_WAITING_NORMAL * normal_wait_sum)
        travel_cost = c.GAMMA_DISTANCE * self.total_distance

        self.total_cost = miss_penalty + response_cost + travel_cost
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
