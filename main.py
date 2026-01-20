import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Callable, Dict, Optional


# ==================== 參數配置 ====================
@dataclass
class MCSConfig:
    """Mobile Charging Station (MCS) 參數配置"""
    SPEED: float = (12 * 3.6) / 60.0    # km/min (原速度 12 m/s)
    CAPACITY: float = 270.0             # kWh 電池容量
    POWER_FAST: float = 250.0           # kW (Super Fast Charging)
    POWER_SLOW: float = 11.0            # kW (AC Slow Charging)


@dataclass
class UAVConfig:
    """Unmanned Aerial Vehicle (UAV) 參數配置"""
    SPEED: float = 60.0 / 60.0          # km/min (60 km/h)
    CAPACITY: float = 1000000.0         # kWh 電池容量
    POWER_FAST: float = 50.0            # kW 快充功率
    MAX_PAYLOAD: float = 50.0           # kWh 最大載電量


@dataclass
class ProblemConfig:
    """問題實例參數配置"""
    RANDOM_SEED: int = 42               # 隨機種子


@dataclass
class ALNSConfig:
    """ALNS 求解器參數配置"""
    # Destroy 參數
    DESTROY_MIN: int = 2                # 最小移除節點數
    DESTROY_MAX: int = 8                # 最大移除節點數
    SHAW_DISTANCE_WEIGHT: float = 0.4   # Shaw removal: 距離權重
    SHAW_TIME_WEIGHT: float = 0.4       # Shaw removal: 時間權重
    SHAW_TYPE_WEIGHT: float = 0.2       # Shaw removal: 類型權重
    
    # Adaptive weights 參數
    SIGMA_1: float = 33.0               # 發現新 global best
    SIGMA_2: float = 9.0                # 改善 current (接受)
    SIGMA_3: float = 3.0                # 接受但未改善
    REACTION_FACTOR: float = 0.1        # 權重更新反應係數 ρ
    SEGMENT_SIZE: int = 50              # 每幾次迭代更新一次權重
    
    # Simulated Annealing 參數
    INIT_TEMP_RATIO: float = 0.1        # T0 = ratio * initial_cost
    COOLING_RATE: float = 0.9995        # 降溫速率
    MIN_TEMP: float = 0.01              # 最低溫度
    
    # 輸出控制
    LOG_INTERVAL: int = 50              # 每幾次迭代輸出一次 log


# ==================== 節點類別 ====================
@dataclass
class Node:
    """節點資料結構"""
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float
    node_type: str = 'normal'  # 'depot', 'normal', 'urgent'


# ==================== 路徑容器類別 ====================
class Route:
    """單一路徑容器"""
    
    def __init__(self, vehicle_type: str = 'mcs', vehicle_id: int = 0):
        """
        初始化路徑
        
        Args:
            vehicle_type: 'mcs' 或 'uav'
            vehicle_id: 車輛編號
        """
        self.vehicle_type: str = vehicle_type
        self.vehicle_id: int = vehicle_id
        self.nodes: List[Node] = []  # 路徑上的節點序列 (不含 depot)
        self.departure_times: List[float] = []  # 各節點的出發時間
        self.arrival_times: List[float] = []  # 各節點的到達時間
        self.user_waiting_times: List[float] = []  # 各節點的用戶等待時間 (用戶發出請求後等充電完成 = departure - ready)
        self.mcs_waiting_times: List[float] = []  # 各節點的 MCS 等待時間 (MCS 到達後等用戶準備好)
        self.charging_modes: List[str] = []  # 各節點的充電模式 ('FAST' 或 'SLOW')
        self.total_distance: float = 0.0
        self.total_time: float = 0.0
        self.total_user_waiting_time: float = 0.0  # 路徑總用戶等待時間
        self.total_mcs_waiting_time: float = 0.0  # 路徑總 MCS 等待時間
        self.total_demand: float = 0.0
        self.is_feasible: bool = True
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __iter__(self):
        return iter(self.nodes)
    
    def __getitem__(self, index: int) -> Node:
        return self.nodes[index]
    
    def add_node(self, node: Node) -> None:
        """新增節點至路徑尾端"""
        self.nodes.append(node)
        self.total_demand += node.demand
    
    def insert_node(self, index: int, node: Node) -> None:
        """在指定位置插入節點"""
        self.nodes.insert(index, node)
        self.total_demand += node.demand
    
    def remove_node(self, index: int) -> Node:
        """移除指定位置的節點"""
        node = self.nodes.pop(index)
        self.total_demand -= node.demand
        return node
    
    def get_node_ids(self) -> List[int]:
        """取得路徑上所有節點的 ID"""
        return [node.id for node in self.nodes]
    
    def copy(self) -> 'Route':
        """深度複製路徑"""
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


class Solution:
    """解的容器 - 包含多條路徑"""
    
    # ===== 階層式目標函數權重 (適用於 ALNS) =====
    # 設計原則：未服務懲罰 >> 用戶等待時間 >> 距離
    # 目標：最小化用戶等待時間 (用戶發出充電請求後，等待 MCS 到達的時間)
    PENALTY_UNASSIGNED: float = 10000.0   # 未服務節點懲罰 (極大，確保不作弊)
    WEIGHT_WAITING: float = 1.0           # 平均用戶等待時間權重 (主要目標)
    WEIGHT_DISTANCE: float = 0.01         # 距離權重 (tie-breaker，避免繞路)
    
    def __init__(self):
        self.mcs_routes: List[Route] = []  # MCS 路徑列表
        self.uav_routes: List[Route] = []  # UAV 路徑列表
        
        # ===== 目標函數值 =====
        self.total_cost: float = float('inf')
        
        # ===== 統計指標 =====
        self.total_distance: float = 0.0
        self.total_time: float = 0.0
        self.total_waiting_time: float = 0.0
        self.avg_waiting_time: float = 0.0
        self.waiting_customers_count: int = 0  # 有等待的客戶數
        self.avg_waiting_time_with_wait: float = 0.0  # 只計有等待客戶的平均等待時間
        self.coverage_rate: float = 0.0
        self.flexibility_score: float = 0.0
        self.unassigned_nodes: List[Node] = []
        self.total_customers: int = 0
        self.is_feasible: bool = False
    
    def add_mcs_route(self, route: Route) -> None:
        """新增 MCS 路徑"""
        route.vehicle_id = len(self.mcs_routes)
        self.mcs_routes.append(route)
    
    def add_uav_route(self, route: Route) -> None:
        """新增 UAV 路徑"""
        route.vehicle_id = len(self.uav_routes)
        self.uav_routes.append(route)
    
    def get_all_routes(self) -> List[Route]:
        """取得所有路徑"""
        return self.mcs_routes + self.uav_routes
    
    def get_assigned_node_ids(self) -> Set[int]:
        """取得所有已分配節點的 ID"""
        assigned = set()
        for route in self.get_all_routes():
            assigned.update(route.get_node_ids())
        return assigned
    
    def calculate_total_cost(self, total_customers: int = None) -> float:
        """
        計算階層式目標函數值 (適用於 ALNS)
        
        Objective Function (Hierarchical):
            Cost = α × 未服務節點數 + β × 平均用戶等待時間 + γ × 總距離
        
        用戶等待時間定義 (Waiting Until Service Completion):
            - 用戶在 READY_TIME 發出充電請求
            - 用戶等待時間 = 服務完成時間 - READY_TIME = departure_time - ready_time
            - 這包含：(1) 等待 MCS 到達的時間 + (2) 充電服務時間
            - 即使 MCS 比 READY_TIME 早到，用戶仍需等待充電完成
        
        設計原則：
            - α (10000): 極大懲罰，確保 ALNS 不會「丟棄客戶」來作弊
            - β (1.0): 主要最佳化目標 (最小化用戶等待時間)
            - γ (0.01): Tie-breaker，避免為了省 1 分鐘而繞路 50 公里
        
        Args:
            total_customers: 總客戶數 (用於計算覆蓋率)
        
        Returns:
            目標函數值
        """
        all_routes = self.get_all_routes()
        
        # 1. 基本指標
        self.total_distance = sum(r.total_distance for r in all_routes)
        self.total_time = sum(r.total_time for r in all_routes)
        
        # 2. 計算用戶等待時間 (用戶發出請求後等待充電服務完成的時間 = departure_time - ready_time)
        self.total_waiting_time = sum(r.total_user_waiting_time for r in all_routes)
        served_count = sum(len(r.nodes) for r in all_routes)
        self.avg_waiting_time = self.total_waiting_time / served_count if served_count > 0 else 0.0
        
        # 計算有等待的客戶數和只計有等待客戶的平均等待時間
        self.waiting_customers_count = 0
        for r in all_routes:
            for wt in r.user_waiting_times:
                if wt > 0:
                    self.waiting_customers_count += 1
        self.avg_waiting_time_with_wait = (
            self.total_waiting_time / self.waiting_customers_count 
            if self.waiting_customers_count > 0 else 0.0
        )
        
        # 3. 計算覆蓋率
        if total_customers is not None:
            self.total_customers = total_customers
        if self.total_customers > 0:
            self.coverage_rate = served_count / self.total_customers
        else:
            self.coverage_rate = 1.0
        
        # 4. 計算彈性分數
        self.flexibility_score = self._calculate_flexibility_score()
        
        # ===== 階層式目標函數 =====
        # 目標：最小化用戶等待時間
        unassigned_penalty = self.PENALTY_UNASSIGNED * len(self.unassigned_nodes)
        waiting_cost = self.WEIGHT_WAITING * self.avg_waiting_time
        distance_cost = self.WEIGHT_DISTANCE * self.total_distance
        
        self.total_cost = unassigned_penalty + waiting_cost + distance_cost
        
        # 可行性判定
        self.is_feasible = len(self.unassigned_nodes) == 0
        
        return self.total_cost
    
    def _calculate_flexibility_score(self) -> float:
        """
        計算調度彈性分數 (報告指標，非最佳化目標)
        
        使用車輛負載的變異係數 (CV) 來衡量平衡度
        CV = 標準差 / 平均值，越小表示負載越平衡
        
        Returns:
            彈性分數 (變異係數，越小越好)
        """
        all_routes = self.get_all_routes()
        if len(all_routes) <= 1:
            return 0.0
        
        # 計算每條路徑的負載比例 (使用時間作為負載指標)
        loads = [r.total_time for r in all_routes if len(r.nodes) > 0]
        
        if len(loads) <= 1:
            return 0.0
        
        mean_load = np.mean(loads)
        if mean_load == 0:
            return 0.0
        
        std_load = np.std(loads)
        cv = std_load / mean_load  # 變異係數
        
        return cv
    
    def copy(self) -> 'Solution':
        """深度複製解"""
        new_solution = Solution()
        new_solution.mcs_routes = [r.copy() for r in self.mcs_routes]
        new_solution.uav_routes = [r.copy() for r in self.uav_routes]
        new_solution.total_cost = self.total_cost
        new_solution.total_distance = self.total_distance
        new_solution.total_time = self.total_time
        new_solution.total_waiting_time = self.total_waiting_time
        new_solution.avg_waiting_time = self.avg_waiting_time
        new_solution.waiting_customers_count = self.waiting_customers_count
        new_solution.avg_waiting_time_with_wait = self.avg_waiting_time_with_wait
        new_solution.coverage_rate = self.coverage_rate
        new_solution.flexibility_score = self.flexibility_score
        new_solution.total_customers = self.total_customers
        new_solution.unassigned_nodes = self.unassigned_nodes.copy()
        new_solution.is_feasible = self.is_feasible
        return new_solution
    
    def __repr__(self) -> str:
        return (f"Solution(MCS={len(self.mcs_routes)}, UAV={len(self.uav_routes)}, "
                f"cost={self.total_cost:.2f}, coverage={self.coverage_rate:.1%}, "
                f"avg_wait={self.avg_waiting_time:.2f}min)")
    
    def print_summary(self) -> None:
        """輸出解的摘要"""
        print("\n" + "="*60)
        print("解的摘要")
        print("="*60)
        
        # 車輛配置
        print("\n" + "車輛配置")
        print(f"MCS路徑數: {len(self.mcs_routes)}")
        print(f"UAV路徑數: {len(self.uav_routes)}")
        
        # 目標函數分解
        print("\n" + "目標函數 (階層式)")
        print(f"總成本: {self.total_cost:.2f}")
        unassigned_penalty = self.PENALTY_UNASSIGNED * len(self.unassigned_nodes)
        waiting_cost = self.WEIGHT_WAITING * self.avg_waiting_time
        distance_cost = self.WEIGHT_DISTANCE * self.total_distance
        print(f"未服務懲罰: {unassigned_penalty:.2f} ({len(self.unassigned_nodes)} × {self.PENALTY_UNASSIGNED})")
        print(f"等待成本: {waiting_cost:.2f} ({self.avg_waiting_time:.2f} min × {self.WEIGHT_WAITING})")
        print(f"距離成本: {distance_cost:.2f} ({self.total_distance:.2f} km × {self.WEIGHT_DISTANCE})")
        
        # 關鍵指標
        print("\n" + "關鍵指標")
        print(f"平均用戶等待時間 (全部客戶): {self.avg_waiting_time:.2f} 分鐘")
        # print(f"平均用戶等待時間 (有等待客戶): {self.avg_waiting_time_with_wait:.2f} 分鐘 ({self.waiting_customers_count}/{self.total_customers - len(self.unassigned_nodes)} 客戶有等待)")
        print(f"覆蓋率: {self.coverage_rate:.1%} ({self.total_customers - len(self.unassigned_nodes)}/{self.total_customers})")
        print(f"可行解: {'是' if self.is_feasible else '否 (有未服務節點)'}")
        
        # 其他指標
        print("\n" + "其他指標")
        # print(f"總距離: {self.total_distance:.2f} km")
        # print(f"總時間: {self.total_time:.2f} 分鐘")
        print(f"總用戶等待時間: {self.total_waiting_time:.2f} 分鐘")
        # print(f"彈性分數 (CV): {self.flexibility_score:.3f}")
        
        # MCS 充電模式統計
        mcs_fast_count = 0
        mcs_slow_count = 0
        for route in self.mcs_routes:
            for mode in route.charging_modes:
                if mode == 'FAST':
                    mcs_fast_count += 1
                elif mode == 'SLOW':
                    mcs_slow_count += 1
        
        print("\n" + "MCS 充電模式統計")
        print(f"  Fast Charging (250 kW): {mcs_fast_count} 次")
        print(f"  Slow Charging (11 kW): {mcs_slow_count} 次")
        if mcs_fast_count + mcs_slow_count > 0:
            slow_ratio = mcs_slow_count / (mcs_fast_count + mcs_slow_count) * 100
            print(f"慢充比例: {slow_ratio:.1f}%")
        
        # 路徑詳情
        if self.mcs_routes:
            print("\n" + "MCS 路徑詳情")
            for route in self.mcs_routes:
                mode_summary = f"[F:{route.charging_modes.count('FAST')}/S:{route.charging_modes.count('SLOW')}]"
                print(f"{route.vehicle_type.upper()}-{route.vehicle_id + 1}: "
                      f"節點={route.get_node_ids()}, "
                      f"距離={route.total_distance:.1f}km, "
                      f"用戶等待={route.total_user_waiting_time:.1f}min, "
                      f"模式={mode_summary}")
        
        if self.uav_routes:
            print("\n" + "UAV 路徑詳情")
            for route in self.uav_routes:
                print(f"{route.vehicle_type.upper()}-{route.vehicle_id + 1}: "
                      f"節點={route.get_node_ids()}, "
                      f"距離={route.total_distance:.1f}km, "
                      f"用戶等待={route.total_user_waiting_time:.1f}min")
        
        print("\n" + "="*60)


# ==================== 距離矩陣快取 ====================
class DistanceMatrix:
    """
    預計算距離矩陣，避免 ALNS 迭代時重複計算
    - MCS: 曼哈頓距離
    - UAV: 歐幾里得距離
    """
    
    def __init__(self, nodes: List[Node]):
        """
        初始化距離矩陣
        
        Args:
            nodes: 包含 depot 在內的所有節點列表
        """
        n = len(nodes)
        self.n = n
        self.nodes = nodes
        
        # 建立 node.id -> 列表索引 的映射
        self.id_to_idx = {node.id: idx for idx, node in enumerate(nodes)}
        
        # 預計算距離矩陣
        self.manhattan = np.zeros((n, n))  # MCS 使用
        self.euclidean = np.zeros((n, n))  # UAV 使用
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = abs(nodes[i].x - nodes[j].x)
                    dy = abs(nodes[i].y - nodes[j].y)
                    self.manhattan[i, j] = dx + dy
                    self.euclidean[i, j] = np.sqrt(dx**2 + dy**2)
    
    def get_distance(self, node1: Node, node2: Node, vehicle: str = 'mcs') -> float:
        """取得兩節點間距離"""
        idx1 = self.id_to_idx[node1.id]
        idx2 = self.id_to_idx[node2.id]
        if vehicle == 'mcs':
            return self.manhattan[idx1, idx2]
        else:
            return self.euclidean[idx1, idx2]
    
    def get_travel_time(self, node1: Node, node2: Node, vehicle: str, 
                        mcs_speed: float, uav_speed: float) -> float:
        """取得行駛時間 (分鐘)"""
        dist = self.get_distance(node1, node2, vehicle)
        if vehicle == 'mcs':
            return dist / mcs_speed
        else:
            return dist / uav_speed


# ==================== ALNS 求解器 ====================
class ALNSSolver:
    """
    Adaptive Large Neighborhood Search 求解器
    
    特性：
    - 使用 Simulated Annealing 作為接受準則
    - 支援 3 種 destroy operators + 2 種 repair operators
    - Adaptive weight 每 segment 更新一次
    """
    
    def __init__(self, 
                 problem: 'ChargingSchedulingProblem',
                 config: ALNSConfig = None,
                 seed: int = None):
        """
        初始化 ALNS 求解器
        
        Args:
            problem: ChargingSchedulingProblem 實例
            config: ALNS 參數配置
            seed: 隨機種子 (None 表示使用 problem 的種子)
        """
        self.problem = problem
        self.config = config or ALNSConfig()
        
        # 設定隨機種子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 預計算距離矩陣
        self.dist_matrix = DistanceMatrix(problem.nodes)
        
        # 總客戶數 (用於 calculate_total_cost)
        self.total_customers = len(problem.nodes) - 1  # 排除 depot
        
        # Destroy operators
        self.destroy_operators = [
            self._random_removal,
            self._worst_removal,
            self._shaw_removal
        ]
        self.destroy_names = ['Random', 'Worst', 'Shaw']
        
        # Repair operators
        self.repair_operators = [
            self._greedy_insertion,
            self._regret_k_insertion
        ]
        self.repair_names = ['Greedy', 'Regret-2']
        
        # Adaptive weights (初始均等)
        self.destroy_weights = np.ones(len(self.destroy_operators))
        self.repair_weights = np.ones(len(self.repair_operators))
        
        # Operator 統計 (用於 segment 內的 weight update)
        self._reset_segment_stats()
    
    def _reset_segment_stats(self) -> None:
        """重置 segment 統計"""
        self.destroy_scores = np.zeros(len(self.destroy_operators))
        self.destroy_usages = np.zeros(len(self.destroy_operators))
        self.repair_scores = np.zeros(len(self.repair_operators))
        self.repair_usages = np.zeros(len(self.repair_operators))
    
    def _roulette_wheel_selection(self, weights: np.ndarray) -> int:
        """輪盤賭選擇"""
        total = weights.sum()
        if total == 0:
            return random.randint(0, len(weights) - 1)
        probs = weights / total
        return np.random.choice(len(weights), p=probs)
    
    def _get_destroy_size(self) -> int:
        """取得本次迭代的 destroy size"""
        return random.randint(self.config.DESTROY_MIN, self.config.DESTROY_MAX)
    
    # ==================== Destroy Operators ====================
    
    def _random_removal(self, solution: Solution, k: int) -> Tuple[Solution, List[Node]]:
        """
        Random Removal: 隨機移除 k 個客戶
        
        Args:
            solution: 當前解 (會被修改)
            k: 移除數量
        
        Returns:
            (修改後的 solution, 被移除的節點列表)
        """
        removed = []
        all_routes = solution.get_all_routes()
        
        # 收集所有已分配的客戶 (node, route, position)
        candidates = []
        for route in all_routes:
            for pos, node in enumerate(route.nodes):
                candidates.append((node, route, pos))
        
        if len(candidates) == 0:
            return solution, removed
        
        # 隨機選取 k 個 (不超過現有數量)
        k = min(k, len(candidates))
        selected = random.sample(candidates, k)
        
        # 按 position 降序排列，從後往前移除避免 index shift
        selected.sort(key=lambda x: x[2], reverse=True)
        
        for node, route, pos in selected:
            route.remove_node(pos)
            removed.append(node)
        
        # 移除空路徑
        self._remove_empty_routes(solution)
        
        return solution, removed
    
    def _worst_removal(self, solution: Solution, k: int) -> Tuple[Solution, List[Node]]:
        """
        Worst Removal: 移除「邊際成本」最高的 k 個客戶
        邊際成本 = 該節點對總距離的貢獻 + 該節點造成的等待時間
        
        Args:
            solution: 當前解 (會被修改)
            k: 移除數量
        
        Returns:
            (修改後的 solution, 被移除的節點列表)
        """
        removed = []
        
        # 收集所有已分配的客戶並計算邊際成本
        candidates = []  # (marginal_cost, node, route, position)
        
        for route in solution.get_all_routes():
            if len(route.nodes) == 0:
                continue
            
            vehicle = route.vehicle_type
            
            for pos, node in enumerate(route.nodes):
                # 計算移除此節點後的距離變化
                prev_node = self.problem.depot if pos == 0 else route.nodes[pos - 1]
                next_node = self.problem.depot if pos == len(route.nodes) - 1 else route.nodes[pos + 1]
                
                # 原本的距離
                dist_before = (self.dist_matrix.get_distance(prev_node, node, vehicle) +
                               self.dist_matrix.get_distance(node, next_node, vehicle))
                # 移除後的距離
                dist_after = self.dist_matrix.get_distance(prev_node, next_node, vehicle)
                
                # 距離節省 (正值表示移除有利)
                dist_saving = dist_before - dist_after
                
                # 等待時間貢獻
                waiting_cost = route.mcs_waiting_times[pos] if pos < len(route.mcs_waiting_times) else 0.0
                
                # 邊際成本：距離 + 等待 (要移除成本高的)
                # 注意：這裡用「負的節省」作為成本，所以節省越多的反而成本越低
                # 我們要移除的是「造成負擔最大的」，即 dist_saving 越大越該移除
                marginal_cost = dist_saving + waiting_cost * 0.1
                
                candidates.append((marginal_cost, node, route, pos))
        
        if len(candidates) == 0:
            return solution, removed
        
        # 按邊際成本降序排列
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 選取前 k 個
        k = min(k, len(candidates))
        
        # 加入一些隨機性：使用 randomized worst removal
        # 隨機選取 top 2k 中的 k 個
        pool_size = min(2 * k, len(candidates))
        pool = candidates[:pool_size]
        selected = random.sample(pool, k)
        
        # 收集要移除的 (route, pos) 對
        to_remove = [(route, pos, node) for _, node, route, pos in selected]
        
        # 按 route 分組並按 pos 降序排列
        route_removals: Dict[Route, List[Tuple[int, Node]]] = {}
        for route, pos, node in to_remove:
            if route not in route_removals:
                route_removals[route] = []
            route_removals[route].append((pos, node))
        
        for route, items in route_removals.items():
            items.sort(key=lambda x: x[0], reverse=True)
            for pos, node in items:
                if pos < len(route.nodes):  # 安全檢查
                    route.remove_node(pos)
                    removed.append(node)
        
        # 移除空路徑
        self._remove_empty_routes(solution)
        
        return solution, removed
    
    def _shaw_removal(self, solution: Solution, k: int) -> Tuple[Solution, List[Node]]:
        """
        Shaw Removal: 移除相似的 k 個客戶 (空間+時間+類型)
        
        Args:
            solution: 當前解 (會被修改)
            k: 移除數量
        
        Returns:
            (修改後的 solution, 被移除的節點列表)
        """
        removed = []
        
        # 收集所有已分配的客戶
        candidates = []
        for route in solution.get_all_routes():
            for pos, node in enumerate(route.nodes):
                candidates.append((node, route, pos))
        
        if len(candidates) == 0:
            return solution, removed
        
        k = min(k, len(candidates))
        
        # 隨機選取 seed 節點
        seed_idx = random.randint(0, len(candidates) - 1)
        seed_node = candidates[seed_idx][0]
        
        # 計算每個節點與 seed 的 relatedness
        relatedness = []
        for node, route, pos in candidates:
            rel = self._calculate_relatedness(seed_node, node)
            relatedness.append((rel, node, route, pos))
        
        # 按 relatedness 升序排列 (越小越相似)
        relatedness.sort(key=lambda x: x[0])
        
        # 選取前 k 個最相似的
        selected = relatedness[:k]
        
        # 收集要移除的
        to_remove = [(route, pos, node) for _, node, route, pos in selected]
        
        # 按 route 分組並按 pos 降序排列
        route_removals: Dict[Route, List[Tuple[int, Node]]] = {}
        for route, pos, node in to_remove:
            if route not in route_removals:
                route_removals[route] = []
            route_removals[route].append((pos, node))
        
        for route, items in route_removals.items():
            items.sort(key=lambda x: x[0], reverse=True)
            for pos, node in items:
                if pos < len(route.nodes):
                    route.remove_node(pos)
                    removed.append(node)
        
        # 移除空路徑
        self._remove_empty_routes(solution)
        
        return solution, removed
    
    def _calculate_relatedness(self, node1: Node, node2: Node) -> float:
        """
        計算兩節點的 relatedness (越小越相似)
        
        Components:
        - 空間距離 (歐式距離，正規化)
        - 時間窗差異 (due_date 差距)
        - 類型差異 (相同類型 = 0，不同 = 1)
        """
        cfg = self.config
        
        # 空間距離 (使用歐式距離)
        dist = self.dist_matrix.get_distance(node1, node2, vehicle='uav')
        
        # 時間窗差異
        time_diff = abs(node1.due_date - node2.due_date)
        
        # 類型差異
        type_same = 1.0 if node1.node_type == node2.node_type else 0.0
        type_diff = 1.0 - type_same
        
        # 加權組合 (距離和時間正規化到相近尺度)
        relatedness = (cfg.SHAW_DISTANCE_WEIGHT * dist +
                       cfg.SHAW_TIME_WEIGHT * time_diff +
                       cfg.SHAW_TYPE_WEIGHT * type_diff * 100)  # type_diff 放大
        
        return relatedness
    
    def _remove_empty_routes(self, solution: Solution) -> None:
        """移除空路徑並重新編號"""
        solution.mcs_routes = [r for r in solution.mcs_routes if len(r.nodes) > 0]
        solution.uav_routes = [r for r in solution.uav_routes if len(r.nodes) > 0]
        
        # 重新編號
        for i, route in enumerate(solution.mcs_routes):
            route.vehicle_id = i
        for i, route in enumerate(solution.uav_routes):
            route.vehicle_id = i
    
    # ==================== Repair Operators ====================
    
    def _greedy_insertion(self, solution: Solution, removed_nodes: List[Node]) -> Solution:
        """
        Greedy Insertion: 依序將節點插入成本增加最小的位置
        
        Args:
            solution: 當前解 (會被修改)
            removed_nodes: 待插入的節點列表
        
        Returns:
            修改後的 solution
        """
        # 按 due_date 排序 (早的優先插入)
        nodes_to_insert = sorted(removed_nodes, key=lambda n: n.due_date)
        
        for node in nodes_to_insert:
            best_route = None
            best_position = None
            best_cost_increase = float('inf')
            
            # 決定可用的路徑
            # urgent 和 normal 只能由 MCS 服務
            candidate_routes = solution.mcs_routes
            
            # 嘗試插入每條現有路徑的每個位置
            for route in candidate_routes:
                for pos in range(len(route.nodes) + 1):
                    # 使用增量檢查 (O(1) ~ O(n)) 而非完整評估 (O(n))
                    feasible, delta_cost = self.problem.incremental_insertion_check(route, pos, node)
                    
                    if feasible:
                        if delta_cost < best_cost_increase:
                            best_cost_increase = delta_cost
                            best_route = route
                            best_position = pos
            
            # 如果找到可行插入點
            if best_route is not None:
                best_route.insert_node(best_position, node)
                self.problem.evaluate_route(best_route)
            else:
                # 嘗試開啟新路徑
                inserted = self._try_create_new_route(solution, node)
                if not inserted:
                    solution.unassigned_nodes.append(node)
        
        return solution
    
    def _regret_k_insertion(self, solution: Solution, removed_nodes: List[Node], k: int = 2) -> Solution:
        """
        Regret-k Insertion: 使用 regret 值決定插入順序
        
        Regret = (次佳插入成本) - (最佳插入成本)
        Regret 越大的節點越先插入，因為它對位置選擇越敏感
        
        Args:
            solution: 當前解 (會被修改)
            removed_nodes: 待插入的節點列表
            k: regret 計算的候選數量
        
        Returns:
            修改後的 solution
        """
        nodes_to_insert = list(removed_nodes)
        
        while nodes_to_insert:
            best_node = None
            best_route = None
            best_position = None
            best_regret = -float('inf')
            best_cost = float('inf')
            
            for node in nodes_to_insert:
                # 決定可用的路徑
                # urgent 和 normal 只能由 MCS 服務
                candidate_routes = solution.mcs_routes
                
                # 收集所有可行插入位置及其成本
                insertion_options = []  # (cost_increase, route, position)
                
                for route in candidate_routes:
                    for pos in range(len(route.nodes) + 1):
                        # 使用增量檢查 (O(1) ~ O(n)) 而非完整評估 (O(n))
                        feasible, delta_cost = self.problem.incremental_insertion_check(route, pos, node)
                        
                        if feasible:
                            insertion_options.append((delta_cost, route, pos))
                
                # 檢查是否可以開新路徑
                new_route_cost = self._estimate_new_route_cost(node)
                if new_route_cost is not None:
                    insertion_options.append((new_route_cost, None, None))  # None 表示新路徑
                
                if len(insertion_options) == 0:
                    # 無法插入此節點，給予極大 regret 讓它先處理
                    regret = float('inf')
                    if regret > best_regret:
                        best_regret = regret
                        best_node = node
                        best_route = None
                        best_position = None
                        best_cost = float('inf')
                    continue
                
                # 按成本排序
                insertion_options.sort(key=lambda x: x[0])
                
                # 計算 regret
                if len(insertion_options) >= k:
                    regret = insertion_options[k-1][0] - insertion_options[0][0]
                else:
                    regret = insertion_options[-1][0] - insertion_options[0][0]
                
                # 選擇 regret 最大的節點
                if regret > best_regret or (regret == best_regret and insertion_options[0][0] < best_cost):
                    best_regret = regret
                    best_node = node
                    best_cost, best_route, best_position = insertion_options[0]
            
            # 執行最佳插入
            if best_node is None:
                break
            
            nodes_to_insert.remove(best_node)
            
            if best_route is None and best_position is None:
                # 開新路徑
                inserted = self._try_create_new_route(solution, best_node)
                if not inserted:
                    solution.unassigned_nodes.append(best_node)
            elif best_route is not None:
                best_route.insert_node(best_position, best_node)
                self.problem.evaluate_route(best_route)
            else:
                solution.unassigned_nodes.append(best_node)
        
        return solution
    
    def _try_create_new_route(self, solution: Solution, node: Node) -> bool:
        """
        嘗試為節點建立新路徑
        
        Args:
            solution: 解
            node: 要插入的節點
        
        Returns:
            是否成功建立
        """
        # 優先嘗試 MCS
        new_route = Route(vehicle_type='mcs')
        new_route.add_node(node)
        if self.problem.evaluate_route(new_route):
            solution.add_mcs_route(new_route)
            return True
        
        # MCS 無法服務，嘗試派遣 UAV
        uav_route = Route(vehicle_type='uav')
        uav_route.add_node(node)
        if self.problem.evaluate_route(uav_route):
            solution.add_uav_route(uav_route)
            return True
        
        return False
    
    def _estimate_new_route_cost(self, node: Node) -> Optional[float]:
        """
        估計為節點開新路徑的成本（使用等待時間）
        
        Returns:
            成本估計值（等待時間），或 None 表示不可行
        """
        vehicle = 'mcs'
        
        depot = self.problem.depot
        dist = (self.dist_matrix.get_distance(depot, node, vehicle) +
                self.dist_matrix.get_distance(node, depot, vehicle))
        
        # 計算行駛時間
        if vehicle == 'mcs':
            speed = self.problem.mcs.SPEED
            power = self.problem.mcs.POWER_SLOW if node.node_type == 'normal' else self.problem.mcs.POWER_FAST
        else:
            speed = self.problem.uav.SPEED
            power = self.problem.uav.POWER_FAST
        
        travel_time = self.dist_matrix.get_distance(depot, node, vehicle) / speed
        service_start = max(travel_time, node.ready_time)
        
        if service_start > node.due_date:
            return None
        
        # 計算充電時間
        charging_time = (node.demand / power) * 60.0
        departure_time = service_start + charging_time
        
        if departure_time > node.due_date:
            return None
        
        # 返回等待時間（與目標函數一致）
        user_waiting_time = departure_time - node.ready_time
        return user_waiting_time
    
    # ==================== Adaptive Weight Update ====================
    
    def _update_weights(self) -> None:
        """更新 operator 權重 (每 segment 呼叫一次)"""
        rho = self.config.REACTION_FACTOR
        
        # 更新 destroy weights
        for i in range(len(self.destroy_operators)):
            if self.destroy_usages[i] > 0:
                avg_score = self.destroy_scores[i] / self.destroy_usages[i]
            else:
                avg_score = 0.0
            self.destroy_weights[i] = (1 - rho) * self.destroy_weights[i] + rho * avg_score
        
        # 更新 repair weights
        for i in range(len(self.repair_operators)):
            if self.repair_usages[i] > 0:
                avg_score = self.repair_scores[i] / self.repair_usages[i]
            else:
                avg_score = 0.0
            self.repair_weights[i] = (1 - rho) * self.repair_weights[i] + rho * avg_score
        
        # 確保權重不為零
        self.destroy_weights = np.maximum(self.destroy_weights, 0.1)
        self.repair_weights = np.maximum(self.repair_weights, 0.1)
        
        # 重置統計
        self._reset_segment_stats()
    
    # ==================== Main Solve Loop ====================
    
    def solve(self, 
              initial_solution: Solution, 
              max_iters: int = 1000,
              time_limit_sec: float = None) -> Solution:
        """
        執行 ALNS 求解
        
        Args:
            initial_solution: 初始解 (通常來自 greedy_construction)
            max_iters: 最大迭代次數
            time_limit_sec: 時間限制 (秒)，None 表示不限制
        
        Returns:
            找到的最佳解
        """
        start_time = time.time()
        
        # 初始化
        current_solution = initial_solution.copy()
        best_solution = initial_solution.copy()
        
        # 確保初始解已計算成本
        current_solution.calculate_total_cost(self.total_customers)
        best_solution.calculate_total_cost(self.total_customers)
        
        # SA 溫度初始化
        temperature = self.config.INIT_TEMP_RATIO * max(current_solution.total_cost, 1.0)
        
        print("\n" + "="*80)
        print("開始 ALNS 求解")
        print("="*80)
        print(f"初始解成本: {initial_solution.total_cost:.2f}")
        print(f"初始溫度: {temperature:.4f}")
        print(f"最大迭代: {max_iters}")
        if time_limit_sec:
            print(f"時間限制: {time_limit_sec} 秒")
        print("-"*80)
        print(f"{'Iter':>6} | {'Best':>10} | {'Current':>10} | {'Avg Wait':>8} | "
              f"{'Coverage':>8} | {'Unassigned':>10} | {'Destroy':>8} | {'Repair':>8} | {'Accept':>6}")
        print("-"*80)
        
        for iteration in range(1, max_iters + 1):
            # 檢查時間限制
            if time_limit_sec and (time.time() - start_time) > time_limit_sec:
                print(f"\n達到時間限制 ({time_limit_sec}s)，提前終止")
                break
            
            # 選擇 operators
            destroy_idx = self._roulette_wheel_selection(self.destroy_weights)
            repair_idx = self._roulette_wheel_selection(self.repair_weights)
            
            destroy_op = self.destroy_operators[destroy_idx]
            repair_op = self.repair_operators[repair_idx]
            
            # 記錄使用次數
            self.destroy_usages[destroy_idx] += 1
            self.repair_usages[repair_idx] += 1
            
            # 複製當前解進行操作
            new_solution = current_solution.copy()
            new_solution.unassigned_nodes = []  # 清空，destroy/repair 會重新填入
            
            # Destroy
            k = self._get_destroy_size()
            new_solution, removed_nodes = destroy_op(new_solution, k)
            
            # Repair
            new_solution = repair_op(new_solution, removed_nodes)
            
            # 重新評估所有路徑
            for route in new_solution.get_all_routes():
                self.problem.evaluate_route(route)
            
            # 計算成本
            new_solution.calculate_total_cost(self.total_customers)
            
            # 計算分數和接受決策
            score = 0.0
            accepted = False
            new_global_best = False
            
            # 檢查是否為新 global best
            if new_solution.total_cost < best_solution.total_cost:
                best_solution = new_solution.copy()
                score = self.config.SIGMA_1
                accepted = True
                new_global_best = True
            # 檢查是否改善 current
            elif new_solution.total_cost < current_solution.total_cost:
                score = self.config.SIGMA_2
                accepted = True
            # SA 接受準則
            else:
                delta = new_solution.total_cost - current_solution.total_cost
                if temperature > self.config.MIN_TEMP:
                    accept_prob = np.exp(-delta / temperature)
                    if random.random() < accept_prob:
                        score = self.config.SIGMA_3
                        accepted = True
            
            # 更新 current solution
            if accepted:
                current_solution = new_solution.copy()
            
            # 累計分數
            self.destroy_scores[destroy_idx] += score
            self.repair_scores[repair_idx] += score
            
            # 降溫
            temperature *= self.config.COOLING_RATE
            temperature = max(temperature, self.config.MIN_TEMP)
            
            # 定期輸出 log
            if iteration % self.config.LOG_INTERVAL == 0:
                accept_str = "Y*" if new_global_best else ("Y" if accepted else "N")
                print(f"{iteration:>6} | {best_solution.total_cost:>10.2f} | "
                      f"{current_solution.total_cost:>10.2f} | "
                      f"{current_solution.avg_waiting_time:>8.2f} | "
                      f"{current_solution.coverage_rate:>7.1%} | "
                      f"{len(current_solution.unassigned_nodes):>10} | "
                      f"{self.destroy_names[destroy_idx]:>8} | "
                      f"{self.repair_names[repair_idx]:>8} | "
                      f"{accept_str:>6}")
            
            # 每 segment 更新權重
            if iteration % self.config.SEGMENT_SIZE == 0:
                self._update_weights()
        
        elapsed_time = time.time() - start_time
        
        print("-"*80)
        print(f"ALNS 完成! 耗時 {elapsed_time:.2f} 秒")
        print(f"最佳成本: {best_solution.total_cost:.2f}")
        print(f"覆蓋率: {best_solution.coverage_rate:.1%}")
        print(f"平均等待時間: {best_solution.avg_waiting_time:.2f} 分鐘")
        print("="*80)
        
        # 輸出 operator 權重統計
        print("\n【Operator 最終權重】")
        print("  Destroy operators:")
        for i, name in enumerate(self.destroy_names):
            print(f"    {name}: {self.destroy_weights[i]:.3f}")
        print("  Repair operators:")
        for i, name in enumerate(self.repair_names):
            print(f"    {name}: {self.repair_weights[i]:.3f}")
        
        return best_solution


# ==================== 視覺化函式 ====================
def plot_routes(solution: Solution, problem: 'ChargingSchedulingProblem', 
                save_path: str = 'routes.png') -> None:
    """
    繪製 MCS 和 UAV 路徑
    
    Args:
        solution: 解
        problem: 問題實例
        save_path: 儲存路徑
    """
    plt.figure(figsize=(12, 10))
    
    # 定義顏色
    mcs_colors = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(solution.mcs_routes), 1)))
    uav_colors = plt.cm.Reds(np.linspace(0.4, 0.9, max(len(solution.uav_routes), 1)))
    
    depot = problem.depot
    
    # 繪製 MCS 路徑
    for i, route in enumerate(solution.mcs_routes):
        if len(route.nodes) == 0:
            continue
        
        # 收集路徑點
        xs = [depot.x] + [n.x for n in route.nodes] + [depot.x]
        ys = [depot.y] + [n.y for n in route.nodes] + [depot.y]
        
        plt.plot(xs, ys, 'o-', color=mcs_colors[i], linewidth=2, 
                 markersize=8, label=f'MCS-{route.vehicle_id + 1}')
    
    # 繪製 UAV 路徑 (虛線表示飛行)
    for i, route in enumerate(solution.uav_routes):
        if len(route.nodes) == 0:
            continue
        
        xs = [depot.x] + [n.x for n in route.nodes] + [depot.x]
        ys = [depot.y] + [n.y for n in route.nodes] + [depot.y]
        
        plt.plot(xs, ys, '^--', color=uav_colors[i], linewidth=2, 
                 markersize=10, label=f'UAV-{route.vehicle_id + 1}')
    
    # 標記節點類型
    for node in problem.nodes:
        if node.node_type == 'depot':
            plt.scatter(node.x, node.y, c='green', marker='*', s=300, zorder=5, edgecolors='black')
            plt.annotate('Depot', (node.x, node.y), textcoords="offset points", 
                        xytext=(10, 10), fontsize=10, fontweight='bold')
        elif node.node_type == 'urgent':
            plt.scatter(node.x, node.y, c='orange', marker='s', s=100, zorder=4, edgecolors='black')
    
    # 標記未服務節點
    for node in solution.unassigned_nodes:
        plt.scatter(node.x, node.y, c='gray', marker='x', s=150, zorder=5, linewidths=3)
    
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title(f'Solution Routes (Cost={solution.total_cost:.2f}, '
              f'Avg Wait={solution.avg_waiting_time:.2f}min)', fontsize=14)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"路徑圖已儲存至: {save_path}")


# ==================== 主程式類別 ====================
class ChargingSchedulingProblem:
    """充電排程問題"""
    
    def __init__(self, 
                 mcs_config: MCSConfig = None,
                 uav_config: UAVConfig = None,
                 problem_config: ProblemConfig = None):
        
        self.mcs = mcs_config or MCSConfig()
        self.uav = uav_config or UAVConfig()
        self.config = problem_config or ProblemConfig()
        
        self.nodes: List[Node] = []
        self.depot: Node = None
        self.urgent_indices: Set[int] = set()
        
        # 設定隨機種子
        random.seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
    
    def load_data(self, filepath: str) -> None:
        """讀取 CSV 資料"""
        df = pd.read_csv(filepath)
        
        # 檢查是否有 NODE_TYPE 欄位
        has_node_type = 'NODE_TYPE' in df.columns
        
        for idx, row in df.iterrows():
            # 讀取節點類型 (若有)
            if has_node_type:
                node_type = str(row['NODE_TYPE']).strip().lower()
            else:
                node_type = 'normal'
            
            node = Node(
                id=int(row['CUST NO.']),
                x=float(row['XCOORD.']),
                y=float(row['YCOORD.']),
                demand=float(row['DEMAND']),
                ready_time=float(row['READY TIME']),
                due_date=float(row['DUE DATE']),
                service_time=float(row['SERVICE TIME']),
                node_type=node_type
            )
            
            # 第一個節點為 depot
            if idx == 0:
                node.node_type = 'depot'
                self.depot = node
            
            self.nodes.append(node)
        
        # 統計節點類型
        type_counts = {}
        for node in self.nodes:
            t = node.node_type
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print(f"載入 {len(self.nodes) - 1} 個節點")
        if has_node_type:
            print(f"節點類型分布: {type_counts}")
    
    def assign_node_types(self, use_csv_types: bool = True) -> None:
        """
        分配節點類型
        
        Args:
            use_csv_types: 是否使用 CSV 中已定義的節點類型。
                          若為 True，則根據已讀取的 node_type 建立索引集合。
                          若為 False，則隨機分配節點類型（舊行為）。
        """
        customer_indices = [i for i in range(1, len(self.nodes))]
        num_customers = len(customer_indices)
        
        if use_csv_types:
            # 根據已讀取的 node_type 建立索引集合
            self.urgent_indices = set(
                i for i in customer_indices if self.nodes[i].node_type == 'urgent'
            )
            normal_indices = [
                i for i in customer_indices if self.nodes[i].node_type == 'normal'
            ]
        else:
            # 舊行為：隨機分配節點類型
            num_urgent = int(num_customers * self.config.URGENT_RATIO)
            self.urgent_indices = set(random.sample(customer_indices, num_urgent))
            
            for idx in self.urgent_indices:
                self.nodes[idx].node_type = 'urgent'
                new_due = self.nodes[idx].ready_time + self.config.TW_URGENT
                self.nodes[idx].due_date = min(new_due, self.depot.due_date)
            
            normal_indices = [
                i for i in customer_indices 
                if i not in self.urgent_indices
            ]
            for idx in normal_indices:
                new_due = self.nodes[idx].ready_time + self.config.TW_NORMAL
                self.nodes[idx].due_date = min(new_due, self.depot.due_date)
        
        print(f"Urgent 節點: {len(self.urgent_indices)} 個")
        print(f"Normal 節點: {len(normal_indices)} 個")
    
    def calculate_distance(self, node1: Node, node2: Node, distance_type: str = 'euclidean') -> float:
        """
        計算兩節點間的距離
        
        Args:
            node1: 起點節點
            node2: 終點節點
            distance_type: 'euclidean' (歐幾里得) 或 'manhattan' (曼哈頓)
        
        Returns:
            距離值
        """
        if distance_type == 'manhattan':
            # 曼哈頓距離 (適用於地面車輛 MCS)
            return abs(node1.x - node2.x) + abs(node1.y - node2.y)
        else:
            # 歐幾里得距離 (適用於 UAV 直線飛行)
            return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def calculate_travel_time(self, node1: Node, node2: Node, vehicle: str = 'mcs') -> float:
        """
        計算行駛時間 (分鐘)
        
        Args:
            node1: 起點節點
            node2: 終點節點
            vehicle: 'mcs' (地面車) 或 'uav' (無人機)
        
        Returns:
            行駛時間 (分鐘)
        """
        if vehicle == 'mcs':
            # MCS 使用曼哈頓距離
            distance = self.calculate_distance(node1, node2, distance_type='manhattan')
            return distance / self.mcs.SPEED
        else:
            # UAV 使用歐幾里得距離
            distance = self.calculate_distance(node1, node2, distance_type='euclidean')
            return distance / self.uav.SPEED
    
    def calculate_charging_time(self, energy_kwh: float, power_kw: float) -> float:
        """計算充電時間 (分鐘)"""
        return (energy_kwh / power_kw) * 60.0
    
    def evaluate_route(self, route: Route) -> bool:
        """
        評估並更新路徑的時間、距離、可行性
        
        實現 Multi-mode MCS 充電模式選擇:
        - UAV: 永遠使用 POWER_FAST
        - MCS + Urgent: 必須使用 POWER_FAST
        - MCS + Normal: 優先嘗試 POWER_SLOW，若不可行則改用 POWER_FAST
        
        Args:
            route: 要評估的路徑
        
        Returns:
            是否為可行路徑
        """
        if len(route.nodes) == 0:
            route.is_feasible = True
            route.total_distance = 0.0
            route.total_time = 0.0
            route.total_waiting_time = 0.0
            route.arrival_times = []
            route.departure_times = []
            route.user_waiting_times = []
            route.mcs_waiting_times = []
            route.charging_modes = []
            return True
        
        vehicle = route.vehicle_type
        capacity = self.mcs.CAPACITY if vehicle == 'mcs' else self.uav.MAX_PAYLOAD
        
        # 檢查載重
        if route.total_demand > capacity:
            route.is_feasible = False
            return False
        
        # 計算時間與距離
        route.arrival_times = []
        route.departure_times = []
        route.user_waiting_times = []
        route.mcs_waiting_times = []
        route.charging_modes = []
        route.total_distance = 0.0
        route.total_user_waiting_time = 0.0
        route.total_mcs_waiting_time = 0.0
        
        current_time = 0.0  # 從 depot 出發時間
        prev_node = self.depot
        
        for i, node in enumerate(route.nodes):
            # 計算行駛距離與時間
            travel_time = self.calculate_travel_time(prev_node, node, vehicle)
            if vehicle == 'mcs':
                distance = self.calculate_distance(prev_node, node, 'manhattan')
            else:
                distance = self.calculate_distance(prev_node, node, 'euclidean')
            
            route.total_distance += distance
            arrival_time = current_time + travel_time
            
            # 檢查是否能在 due_date 前到達
            if arrival_time > node.due_date:
                route.is_feasible = False
                return False
            
            # ===== 等待時間計算 =====
            # user_waiting_time 將在計算 departure_time 後再計算
            
            # MCS 等待時間: MCS 到達後，等待用戶準備好 (READY_TIME)
            # 若 MCS 在 READY_TIME 之前到達，MCS 需要等待
            mcs_waiting_time = max(0.0, node.ready_time - arrival_time)
            
            service_start = max(arrival_time, node.ready_time)
            
            # ===== 動態充電模式選擇 =====
            if vehicle == 'uav':
                # UAV: 永遠使用快充
                charging_power = self.uav.POWER_FAST
                charging_mode = 'FAST'
            else:
                # MCS: 根據節點類型決定充電模式
                if node.node_type == 'urgent':
                    # Urgent 節點: 必須使用快充
                    charging_power = self.mcs.POWER_FAST
                    charging_mode = 'FAST'
                else:
                    # Normal 節點: 優先嘗試慢充
                    # 計算慢充所需時間
                    slow_service_time = self.calculate_charging_time(node.demand, self.mcs.POWER_SLOW)
                    slow_departure_time = service_start + slow_service_time
                    
                    # 檢查慢充可行性
                    slow_feasible = True
                    
                    # 條件 1: 慢充是否會超過當前節點的 due_date?
                    if slow_departure_time > node.due_date:
                        slow_feasible = False
                    
                    # 條件 2: 慢充是否會導致下一個節點無法在 deadline 前到達?
                    if slow_feasible and i + 1 < len(route.nodes):
                        next_node = route.nodes[i + 1]
                        travel_to_next = self.calculate_travel_time(node, next_node, vehicle)
                        arrival_at_next = slow_departure_time + travel_to_next
                        if arrival_at_next > next_node.due_date:
                            slow_feasible = False
                    
                    # 條件 3: 慢充是否會導致無法及時返回 depot?
                    if slow_feasible and i == len(route.nodes) - 1:
                        travel_to_depot = self.calculate_travel_time(node, self.depot, vehicle)
                        return_time = slow_departure_time + travel_to_depot
                        if return_time > self.depot.due_date:
                            slow_feasible = False
                    
                    # 決定充電模式: Normal 強制使用 Slow Charging
                    # 若 Slow 不可行，則此路徑無法服務該節點
                    if slow_feasible:
                        charging_power = self.mcs.POWER_SLOW
                        charging_mode = 'SLOW'
                    else:
                        # Normal 節點強制 Slow，無法 fallback 到 Fast
                        route.is_feasible = False
                        return False
            
            # 計算實際服務時間
            service_time = self.calculate_charging_time(node.demand, charging_power)
            departure_time = service_start + service_time
            
            # ===== 檢查充電是否在 due_date 內完成 =====
            # 所有節點類型都必須在 due_date 前完成充電
            if departure_time > node.due_date:
                route.is_feasible = False
                return False
            
            # ===== 計算用戶等待時間 (含充電時間) =====
            # user_waiting_time = 用戶從發出請求到充電完成的總時間
            user_waiting_time = departure_time - node.ready_time
            
            route.arrival_times.append(arrival_time)
            route.departure_times.append(departure_time)
            route.user_waiting_times.append(user_waiting_time)
            route.mcs_waiting_times.append(mcs_waiting_time)
            route.charging_modes.append(charging_mode)
            route.total_user_waiting_time += user_waiting_time
            route.total_mcs_waiting_time += mcs_waiting_time
            
            current_time = departure_time
            prev_node = node
        
        # 返回 depot
        travel_time_back = self.calculate_travel_time(prev_node, self.depot, vehicle)
        if vehicle == 'mcs':
            distance_back = self.calculate_distance(prev_node, self.depot, 'manhattan')
        else:
            distance_back = self.calculate_distance(prev_node, self.depot, 'euclidean')
        
        route.total_distance += distance_back
        route.total_time = current_time + travel_time_back
        
        # 檢查是否能在 depot 的 due_date 前返回
        if route.total_time > self.depot.due_date:
            route.is_feasible = False
            return False
        
        # 檢查電量 (簡化：假設每公里消耗固定電量)
        # MCS: 假設 0.5 kWh/km, UAV: 假設 0.3 kWh/km
        if vehicle == 'mcs':
            energy_consumed = route.total_distance * 0.5
            if energy_consumed > self.mcs.CAPACITY:
                route.is_feasible = False
                return False
        else:
            energy_consumed = route.total_distance * 0.3
            if energy_consumed > self.uav.CAPACITY:
                route.is_feasible = False
                return False
        
        route.is_feasible = True
        return True
    
    def try_insert_node(self, route: Route, node: Node, position: int = None) -> bool:
        """
        嘗試在路徑的指定位置插入節點
        
        插入準則：最小化用戶等待時間（與目標函數一致）
        
        Args:
            route: 目標路徑
            node: 要插入的節點
            position: 插入位置 (None 表示嘗試所有位置找最佳)
        
        Returns:
            是否成功插入
        """
        if position is not None:
            # 在指定位置插入
            route.insert_node(position, node)
            if self.evaluate_route(route):
                return True
            else:
                route.remove_node(position)
                return False
        
        # 嘗試所有可能的插入位置，找到用戶等待時間最小的位置
        best_position = None
        best_waiting_time = float('inf')
        
        for pos in range(len(route.nodes) + 1):
            route.insert_node(pos, node)
            if self.evaluate_route(route):
                # 以用戶等待時間為主要準則，距離作為 tie-breaker
                if route.total_user_waiting_time < best_waiting_time:
                    best_waiting_time = route.total_user_waiting_time
                    best_position = pos
            route.remove_node(pos)
        
        if best_position is not None:
            route.insert_node(best_position, node)
            self.evaluate_route(route)
            return True
        
        return False
    
    def incremental_insertion_check(self, route: Route, pos: int, node: Node) -> Tuple[bool, float]:
        """
        增量插入可行性檢查與成本計算 (O(1) ~ O(n))
        
        比起完整的 evaluate_route()，此方法只計算插入節點後的局部變化，
        並透過前向傳播檢查時間窗約束。
        
        Args:
            route: 目標路徑 (不會被修改)
            pos: 插入位置 (0 = 路徑開頭)
            node: 要插入的節點
        
        Returns:
            (is_feasible, delta_user_waiting_time)
            若不可行，delta_user_waiting_time = float('inf')
        """
        vehicle = route.vehicle_type
        
        # ===== 1. 載重檢查 O(1) =====
        if vehicle == 'mcs':
            capacity = self.mcs.CAPACITY
        else:
            capacity = self.uav.MAX_PAYLOAD
        
        if route.total_demand + node.demand > capacity:
            return False, float('inf')
        
        # ===== 2. 計算到達新節點的時間 O(1) =====
        if pos == 0:
            # 從 depot 出發
            prev_departure_time = 0.0
            prev_node = self.depot
        else:
            prev_departure_time = route.departure_times[pos - 1]
            prev_node = route.nodes[pos - 1]
        
        travel_time_to_new = self.calculate_travel_time(prev_node, node, vehicle)
        arrival_at_new = prev_departure_time + travel_time_to_new
        
        # 檢查是否能在 due_date 前到達
        if arrival_at_new > node.due_date:
            return False, float('inf')
        
        # ===== 3. 決定充電模式並計算服務時間 O(1) =====
        service_start_new = max(arrival_at_new, node.ready_time)
        
        if vehicle == 'uav':
            charging_power = self.uav.POWER_FAST
        else:
            if node.node_type == 'urgent':
                charging_power = self.mcs.POWER_FAST
            else:
                # Normal 節點強制使用 Slow Charging
                charging_power = self.mcs.POWER_SLOW
        
        charging_time_new = self.calculate_charging_time(node.demand, charging_power)
        departure_from_new = service_start_new + charging_time_new
        
        # 檢查服務是否能在 due_date 前完成
        if departure_from_new > node.due_date:
            return False, float('inf')
        
        # ===== 4. 計算新節點的用戶等待時間 O(1) =====
        new_node_waiting = departure_from_new - node.ready_time
        
        # ===== 5. 計算對後續節點的時間偏移 O(n) worst case =====
        # 需要檢查插入後，後續節點是否仍能滿足時間窗約束
        
        if pos < len(route.nodes):
            # 有後續節點需要檢查
            next_node = route.nodes[pos]
            
            # 計算到下一個節點的行駛時間
            travel_time_to_next = self.calculate_travel_time(node, next_node, vehicle)
            new_arrival_at_next = departure_from_new + travel_time_to_next
            
            # 原本的到達時間
            old_arrival_at_next = route.arrival_times[pos]
            
            # 時間偏移量 (正值表示延後)
            time_shift = new_arrival_at_next - old_arrival_at_next
            
            # 前向傳播檢查可行性
            feasible, delta_successor_waiting = self._forward_propagate_check(
                route, pos, time_shift
            )
            
            if not feasible:
                return False, float('inf')
        else:
            # 插入在路徑末端
            delta_successor_waiting = 0.0
            
            # 檢查是否能及時返回 depot
            travel_time_to_depot = self.calculate_travel_time(node, self.depot, vehicle)
            return_time = departure_from_new + travel_time_to_depot
            
            if return_time > self.depot.due_date:
                return False, float('inf')
        
        # ===== 6. 計算總 delta cost =====
        delta_cost = new_node_waiting + delta_successor_waiting
        
        return True, delta_cost
    
    def _forward_propagate_check(self, route: Route, start_pos: int, 
                                  time_shift: float) -> Tuple[bool, float]:
        """
        前向傳播時間偏移檢查
        
        檢查從 start_pos 開始，若到達時間延後 time_shift，後續節點是否仍可行。
        同時計算用戶等待時間的變化量。
        
        Args:
            route: 路徑
            start_pos: 開始檢查的位置 (該位置的節點會受到影響)
            time_shift: 時間偏移量 (正值 = 延後)
        
        Returns:
            (is_feasible, delta_waiting_time)
        """
        if time_shift <= 0:
            # 提前或不變，一定可行，不影響等待時間
            return True, 0.0
        
        delta_waiting = 0.0
        current_shift = time_shift
        vehicle = route.vehicle_type
        
        for i in range(start_pos, len(route.nodes)):
            node = route.nodes[i]
            old_arrival = route.arrival_times[i]
            new_arrival = old_arrival + current_shift
            
            # 檢查是否能在 due_date 前到達
            if new_arrival > node.due_date:
                return False, float('inf')
            
            # 計算新的服務開始時間
            old_service_start = max(old_arrival, node.ready_time)
            new_service_start = max(new_arrival, node.ready_time)
            
            # 充電時間不變 (demand 和 power 不變)
            charging_time = route.departure_times[i] - old_service_start
            
            new_departure = new_service_start + charging_time
            
            # 檢查服務是否能在 due_date 前完成
            if new_departure > node.due_date:
                return False, float('inf')
            
            # 計算等待時間變化
            old_waiting = route.user_waiting_times[i]
            new_waiting = new_departure - node.ready_time
            delta_waiting += (new_waiting - old_waiting)
            
            # 更新時間偏移量 (考慮等待時間可能吸收部分偏移)
            old_departure = route.departure_times[i]
            current_shift = new_departure - old_departure
            
            # 如果偏移量已被吸收，後續節點不受影響
            if current_shift <= 0:
                break
        
        # 檢查是否能及時返回 depot
        if current_shift > 0:
            old_return_time = route.total_time
            new_return_time = old_return_time + current_shift
            if new_return_time > self.depot.due_date:
                return False, float('inf')
        
        return True, delta_waiting

    
    def greedy_construction(self) -> Solution:
        """
        Greedy Construction Heuristic - Earliest Due Date First + Minimum Waiting Time Insertion
        
        目標：最小化用戶等待時間
        
        策略：
        1. 將所有客戶依據 Due Date 由早到晚排序
        2. 對每個客戶，遍歷所有候選路徑的所有可行插入位置
        3. 選擇「等待時間增加最少」的 (路徑, 位置) 組合
        4. 如果現有車輛都塞不進去，就開啟一輛新車
        
        Returns:
            建構出的初始解
        """
        solution = Solution()
        
        # 取得所有客戶節點 (排除 depot)
        customers = [node for node in self.nodes if node.node_type != 'depot']
        
        # 按 Due Date 由早到晚排序
        customers.sort(key=lambda n: n.due_date)
        
        print("\n開始 Greedy Construction (EDD First + Min Waiting Time)...")
        print(f"待分配客戶數: {len(customers)}")
        
        for customer in customers:
            inserted = False
            
            # urgent 和 normal 只能由 MCS 服務
            candidate_routes = solution.mcs_routes
            
            # ===== 核心改動：比較所有路徑，選擇等待時間增加最少的 =====
            best_route = None
            best_position = None
            best_waiting_increase = float('inf')
            
            for route in candidate_routes:
                # 使用增量檢查 (O(1) ~ O(n)) 而非完整評估 (O(n))
                for pos in range(len(route.nodes) + 1):
                    feasible, delta_cost = self.incremental_insertion_check(route, pos, customer)
                    if feasible:
                        if delta_cost < best_waiting_increase:
                            best_waiting_increase = delta_cost
                            best_route = route
                            best_position = pos
            
            # 如果找到可行插入點
            if best_route is not None:
                best_route.insert_node(best_position, customer)
                self.evaluate_route(best_route)  # 完整評估以更新路徑元數據
                inserted = True
            
            # 如果無法插入現有路徑，開啟新車
            if not inserted:
                # 優先嘗試 MCS
                new_route = Route(vehicle_type='mcs')
                new_route.add_node(customer)
                if self.evaluate_route(new_route):
                    solution.add_mcs_route(new_route)
                    inserted = True
                else:
                    # MCS 無法服務，嘗試派遣 UAV
                    uav_route = Route(vehicle_type='uav')
                    uav_route.add_node(customer)
                    if self.evaluate_route(uav_route):
                        solution.add_uav_route(uav_route)
                        inserted = True
                    else:
                        solution.unassigned_nodes.append(customer)
        
        # 計算總成本 (傳入總客戶數以計算覆蓋率)
        solution.calculate_total_cost(total_customers=len(customers))
        solution.is_feasible = len(solution.unassigned_nodes) == 0
        
        print(f"\nGreedy Construction 完成!")
        print(f"  MCS 路徑數: {len(solution.mcs_routes)}")
        print(f"  UAV 路徑數: {len(solution.uav_routes)}")
        print(f"  未分配節點: {len(solution.unassigned_nodes)}")
        print(f"  覆蓋率: {solution.coverage_rate:.1%}")
        print(f"  平均等待時間: {solution.avg_waiting_time:.2f} 分鐘")
        
        return solution
    
    def plot_nodes(self, save_path: str = 'node_distribution.png') -> None:
        """繪製節點分布圖"""
        plt.figure(figsize=(10, 8))
        
        # 分類節點
        normal_nodes = [n for n in self.nodes if n.node_type == 'normal']
        urgent_nodes = [n for n in self.nodes if n.node_type == 'urgent']
        
        # 繪製普通節點 (藍色)
        if normal_nodes:
            plt.scatter([n.x for n in normal_nodes], [n.y for n in normal_nodes],
                       c='blue', marker='o', s=50, label='Normal')
        
        # 繪製 Urgent 節點 (橘色)
        if urgent_nodes:
            plt.scatter([n.x for n in urgent_nodes], [n.y for n in urgent_nodes],
                       c='orange', marker='s', s=80, label='Urgent')
        
        # 繪製 Depot (綠色)
        if self.depot:
            plt.scatter(self.depot.x, self.depot.y,
                       c='green', marker='*', s=200, label='Depot')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('R101 Node Distribution')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"圖片已儲存至: {save_path}")
    
    def print_config(self) -> None:
        """輸出參數配置"""
        # print("\n" + "="*50)
        print("MCS 參數配置:")
        print(f"  速度: {self.mcs.SPEED:.3f} km/min ({self.mcs.SPEED * 60:.1f} km/h)")
        print(f"  電池容量: {self.mcs.CAPACITY} kWh")
        print(f"  快充功率: {self.mcs.POWER_FAST} kW")
        print(f"  慢充功率: {self.mcs.POWER_SLOW} kW")
        print("\nUAV 參數配置:")
        print(f"  速度: {self.uav.SPEED:.3f} km/min ({self.uav.SPEED * 60:.1f} km/h)")
        print(f"  電池容量: {self.uav.CAPACITY} kWh")
        print(f"  快充功率: {self.uav.POWER_FAST} kW")
        print(f"  最大載電量: {self.uav.MAX_PAYLOAD} kWh")
        print("="*80 + "\n")


# ==================== 主程式入口 ====================
def main():
    # 初始化問題
    problem = ChargingSchedulingProblem()
    
    # 輸出參數配置
    problem.print_config()
    
    # 載入資料 (使用 mixed_instance.csv，節點類型已在 CSV 中定義)
    problem.load_data('mixed_instance.csv')
    
    # 分配節點類型 (使用 CSV 中的類型定義)
    problem.assign_node_types(use_csv_types=True)
    
    # 繪製節點分布圖
    problem.plot_nodes()
    
    # 輸出節點資訊
    # print(f"Urgent 節點索引: {sorted(problem.urgent_indices)}")
    
    # ==================== Phase 1: Greedy Construction ====================
    print("\n" + "="*80)
    print("Phase 1: Greedy Construction Heuristic")
    print("="*80)
    
    greedy_solution = problem.greedy_construction()
    greedy_solution.print_summary()
    
    # 儲存 Greedy 解供後續比較
    greedy_cost = greedy_solution.total_cost
    greedy_avg_wait = greedy_solution.avg_waiting_time
    greedy_coverage = greedy_solution.coverage_rate
    greedy_mcs_routes = len(greedy_solution.mcs_routes)
    greedy_uav_routes = len(greedy_solution.uav_routes)
    
    # 繪製 Greedy 解路徑
    plot_routes(greedy_solution, problem, save_path='routes_greedy.png')
    
    # ==================== Phase 2: ALNS Optimization ====================
    # print("\n" + "="*80)
    # print("Phase 2: ALNS Optimization")
    # print("="*80)
    
    # # 初始化 ALNS 求解器
    # alns_config = ALNSConfig()
    # alns = ALNSSolver(problem, config=alns_config, seed=problem.config.RANDOM_SEED)
    
    # # 執行 ALNS (可調整參數)
    # best_solution = alns.solve(
    #     initial_solution=greedy_solution,
    #     max_iters=5000,       # 迭代次數
    #     time_limit_sec=120    # 時間限制 (秒)
    # )
    
    # # 輸出 ALNS 最佳解摘要
    # print("\n" + "="*80)
    # print("ALNS 最佳解")
    # print("="*80)
    # best_solution.print_summary()
    
    # # 繪製 ALNS 最佳解路徑
    # plot_routes(best_solution, problem, save_path='routes_alns.png')
    
    # # ==================== 比較輸出 ====================
    # print("\n" + "="*80)
    # print("Greedy vs ALNS 比較")
    # print("="*80)
    
    # print(f"\n{'指標':<20} | {'Greedy':>15} | {'ALNS':>15} | {'改善幅度':>15}")
    # print("-" * 72)
    
    # cost_improve = (greedy_cost - best_solution.total_cost) / greedy_cost * 100 if greedy_cost > 0 else 0
    # print(f"{'總成本':<20} | {greedy_cost:>15.2f} | {best_solution.total_cost:>15.2f} | "
    #       f"{cost_improve:>14.1f}%")
    
    # wait_improve = (greedy_avg_wait - best_solution.avg_waiting_time) / greedy_avg_wait * 100 if greedy_avg_wait > 0 else 0
    # print(f"{'平均等待時間 (min)':<20} | {greedy_avg_wait:>15.2f} | {best_solution.avg_waiting_time:>15.2f} | "
    #       f"{wait_improve:>14.1f}%")
    
    # print(f"{'覆蓋率':<20} | {greedy_coverage:>14.1%} | {best_solution.coverage_rate:>14.1%} | "
    #       f"{(best_solution.coverage_rate - greedy_coverage)*100:>14.1f}%")
    
    # print(f"{'MCS 路徑數':<20} | {greedy_mcs_routes:>15} | {len(best_solution.mcs_routes):>15} | "
    #       f"{len(best_solution.mcs_routes) - greedy_mcs_routes:>15}")
    
    # print(f"{'UAV 路徑數':<20} | {greedy_uav_routes:>15} | {len(best_solution.uav_routes):>15} | "
    #       f"{len(best_solution.uav_routes) - greedy_uav_routes:>15}")
    
    # print(f"{'未服務節點數':<20} | {len(greedy_solution.unassigned_nodes):>15} | "
    #       f"{len(best_solution.unassigned_nodes):>15} | "
    #       f"{len(greedy_solution.unassigned_nodes) - len(best_solution.unassigned_nodes):>15}")
    
    # print("-" * 72)
    # print(f"\n圖片已儲存:")
    # print(f"  - 節點分布圖: node_distribution.png")
    # print(f"  - Greedy 路徑: routes_greedy.png")
    # print(f"  - ALNS 路徑: routes_alns.png")


if __name__ == "__main__":
    main()

