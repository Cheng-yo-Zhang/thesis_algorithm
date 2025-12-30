import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
import matplotlib.pyplot as plt
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Set


# ==================== 參數配置 ====================
@dataclass
class MCSConfig:
    """Mobile Charging Station (MCS) 參數配置"""
    SPEED: float = (12 * 3.6) / 60.0   # km/min (原速度 12 m/s)
    CAPACITY: float = 270.0             # kWh 電池容量
    POWER_FAST: float = 250.0           # kW (Super Fast Charging)
    POWER_SLOW: float = 11.0            # kW (AC Slow Charging)


@dataclass
class UAVConfig:
    """Unmanned Aerial Vehicle (UAV) 參數配置"""
    SPEED: float = 60.0 / 60.0          # km/min (60 km/h)
    CAPACITY: float = 15.0              # kWh 電池容量
    POWER_FAST: float = 50.0            # kW 快充功率
    MAX_PAYLOAD: float = 12.0           # kWh 最大載電量


@dataclass
class ProblemConfig:
    """問題實例參數配置"""
    HARD_TO_ACCESS_RATIO: float = 0.10  # Hard-to-Access 節點比例
    URGENT_RATIO: float = 0.20          # Urgent 節點比例
    RANDOM_SEED: int = 42               # 隨機種子


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
    node_type: str = 'normal'  # 'depot', 'normal', 'hard_to_access', 'urgent'


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
        self.waiting_times: List[float] = []  # 各節點的等待時間
        self.total_distance: float = 0.0
        self.total_time: float = 0.0
        self.total_waiting_time: float = 0.0  # 路徑總等待時間
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
        new_route.waiting_times = self.waiting_times.copy()
        new_route.total_distance = self.total_distance
        new_route.total_time = self.total_time
        new_route.total_waiting_time = self.total_waiting_time
        new_route.total_demand = self.total_demand
        new_route.is_feasible = self.is_feasible
        return new_route
    
    def __repr__(self) -> str:
        node_ids = self.get_node_ids()
        return f"Route({self.vehicle_type}-{self.vehicle_id}): {node_ids}, demand={self.total_demand:.1f}"


class Solution:
    """解的容器 - 包含多條路徑"""
    
    # ===== 階層式目標函數權重 (適用於 ALNS) =====
    # 設計原則：未服務懲罰 >> 等待時間 >> 距離
    PENALTY_UNASSIGNED: float = 10000.0   # 未服務節點懲罰 (極大，確保不作弊)
    WEIGHT_WAITING: float = 1.0           # 平均等待時間權重 (主要目標)
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
            Cost = α × 未服務節點數 + β × 平均等待時間 + γ × 總距離
        
        設計原則：
            - α (10000): 極大懲罰，確保 ALNS 不會「丟棄客戶」來作弊
            - β (1.0): 主要最佳化目標
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
        
        # 2. 計算等待時間
        self.total_waiting_time = sum(r.total_waiting_time for r in all_routes)
        served_count = sum(len(r.nodes) for r in all_routes)
        self.avg_waiting_time = self.total_waiting_time / served_count if served_count > 0 else 0.0
        
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
        # 情境 A 防護：未服務節點有極大懲罰，ALNS 絕不會選擇丟棄客戶
        # 情境 B 防護：距離有小權重，繞路 50km 會增加 0.5 成本，不划算
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
        print("📊 解的摘要")
        print("="*60)
        
        # 車輛配置
        print("\n【車輛配置】")
        print(f"  MCS 路徑數: {len(self.mcs_routes)}")
        print(f"  UAV 路徑數: {len(self.uav_routes)}")
        
        # 目標函數分解
        print("\n【目標函數 (階層式)】")
        print(f"  🎯 總成本: {self.total_cost:.2f}")
        unassigned_penalty = self.PENALTY_UNASSIGNED * len(self.unassigned_nodes)
        waiting_cost = self.WEIGHT_WAITING * self.avg_waiting_time
        distance_cost = self.WEIGHT_DISTANCE * self.total_distance
        print(f"  ├─ 未服務懲罰: {unassigned_penalty:.2f} ({len(self.unassigned_nodes)} × {self.PENALTY_UNASSIGNED})")
        print(f"  ├─ 等待成本: {waiting_cost:.2f} ({self.avg_waiting_time:.2f} min × {self.WEIGHT_WAITING})")
        print(f"  └─ 距離成本: {distance_cost:.2f} ({self.total_distance:.2f} km × {self.WEIGHT_DISTANCE})")
        
        # 關鍵指標
        print("\n【關鍵指標】")
        print(f"  平均等待時間: {self.avg_waiting_time:.2f} 分鐘 ← 主要目標")
        print(f"  覆蓋率: {self.coverage_rate:.1%} ({self.total_customers - len(self.unassigned_nodes)}/{self.total_customers})")
        print(f"  可行解: {'✅ 是' if self.is_feasible else '❌ 否 (有未服務節點)'}")
        
        # 其他指標
        print("\n【其他指標】")
        print(f"  總距離: {self.total_distance:.2f} km")
        print(f"  總時間: {self.total_time:.2f} 分鐘")
        print(f"  總等待時間: {self.total_waiting_time:.2f} 分鐘")
        print(f"  彈性分數 (CV): {self.flexibility_score:.3f}")
        
        # 路徑詳情
        if self.mcs_routes:
            print("\n【MCS 路徑詳情】")
            for route in self.mcs_routes:
                print(f"  {route.vehicle_type.upper()}-{route.vehicle_id}: "
                      f"節點={route.get_node_ids()}, "
                      f"距離={route.total_distance:.1f}km, "
                      f"等待={route.total_waiting_time:.1f}min")
        
        if self.uav_routes:
            print("\n【UAV 路徑詳情】")
            for route in self.uav_routes:
                print(f"  {route.vehicle_type.upper()}-{route.vehicle_id}: "
                      f"節點={route.get_node_ids()}, "
                      f"距離={route.total_distance:.1f}km, "
                      f"等待={route.total_waiting_time:.1f}min")
        
        print("\n" + "="*60)


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
        self.hard_to_access_indices: Set[int] = set()
        self.urgent_indices: Set[int] = set()
        
        # 設定隨機種子
        random.seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
    
    def load_data(self, filepath: str) -> None:
        """讀取 CSV 資料"""
        df = pd.read_csv(filepath)
        
        for idx, row in df.iterrows():
            node = Node(
                id=int(row['CUST NO.']),
                x=float(row['XCOORD.']),
                y=float(row['YCOORD.']),
                demand=float(row['DEMAND']),
                ready_time=float(row['READY TIME']),
                due_date=float(row['DUE DATE']),
                service_time=float(row['SERVICE TIME'])
            )
            
            if idx == 0:
                node.node_type = 'depot'
                self.depot = node
            
            self.nodes.append(node)
        
        print(f"載入 {len(self.nodes)} 個節點 (含 depot)")
    
    def assign_node_types(self) -> None:
        """隨機分配節點類型，並為 Urgent 節點縮緊時間窗"""
        customer_indices = [i for i in range(1, len(self.nodes))]
        num_customers = len(customer_indices)
        
        # 隨機選取 Hard-to-Access 節點
        num_hard = int(num_customers * self.config.HARD_TO_ACCESS_RATIO)
        self.hard_to_access_indices = set(random.sample(customer_indices, num_hard))
        
        for idx in self.hard_to_access_indices:
            self.nodes[idx].node_type = 'hard_to_access'
        
        # 從剩餘節點選取 Urgent 節點
        remaining = [i for i in customer_indices if i not in self.hard_to_access_indices]
        num_urgent = int(num_customers * self.config.URGENT_RATIO)
        self.urgent_indices = set(random.sample(remaining, num_urgent))
        
        for idx in self.urgent_indices:
            self.nodes[idx].node_type = 'urgent'
            # 縮緊時間窗：從 Ready Time 開始只有 60 分鐘存活時間
            self.nodes[idx].due_date = self.nodes[idx].ready_time + 60.0
        
        print(f"Hard-to-Access 節點: {len(self.hard_to_access_indices)} 個")
        print(f"Urgent 節點: {len(self.urgent_indices)} 個 (時間窗縮緊為 60 分鐘)")
        print(f"Normal 節點: {num_customers - len(self.hard_to_access_indices) - len(self.urgent_indices)} 個")
    
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
            route.waiting_times = []
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
        route.waiting_times = []  # 新增：記錄等待時間
        route.total_distance = 0.0
        route.total_waiting_time = 0.0  # 新增：總等待時間
        
        current_time = 0.0  # 從 depot 出發時間
        prev_node = self.depot
        
        for node in route.nodes:
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
            
            # 如果提早到達，等待至 ready_time
            # 等待時間 = max(0, ready_time - arrival_time)
            waiting_time = max(0.0, node.ready_time - arrival_time)
            service_start = max(arrival_time, node.ready_time)
            
            # 動態計算充電時間 (根據需求電量和車輛充電功率)
            if vehicle == 'mcs':
                charging_power = self.mcs.POWER_FAST  # MCS 使用快充 250 kW
            else:
                charging_power = self.uav.POWER_FAST  # UAV 使用 50 kW
            service_time = self.calculate_charging_time(node.demand, charging_power)
            departure_time = service_start + service_time
            
            route.arrival_times.append(arrival_time)
            route.departure_times.append(departure_time)
            route.waiting_times.append(waiting_time)  # 記錄等待時間
            route.total_waiting_time += waiting_time  # 累加總等待時間
            
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
        
        # 嘗試所有可能的插入位置，找到可行的位置
        best_position = None
        best_cost = float('inf')
        
        for pos in range(len(route.nodes) + 1):
            route.insert_node(pos, node)
            if self.evaluate_route(route):
                if route.total_distance < best_cost:
                    best_cost = route.total_distance
                    best_position = pos
            route.remove_node(pos)
        
        if best_position is not None:
            route.insert_node(best_position, node)
            self.evaluate_route(route)
            return True
        
        return False
    
    def greedy_construction(self) -> Solution:
        """
        Greedy Construction Heuristic - Earliest Due Date First
        
        1. 將所有客戶依據 Due Date 由早到晚排序
        2. 依序嘗試將客戶插入現有的車輛路徑中
        3. Feasibility Check: 時間窗, 載重, 電量
        4. 如果現有車輛都塞不進去，就開啟一輛新車
        5. hard_to_access 節點強制只能由 UAV 服務
        
        Returns:
            建構出的初始解
        """
        solution = Solution()
        
        # 取得所有客戶節點 (排除 depot)
        customers = [node for node in self.nodes if node.node_type != 'depot']
        
        # 按 Due Date 由早到晚排序
        customers.sort(key=lambda n: n.due_date)
        
        print("\n開始 Greedy Construction (EDD First)...")
        print(f"待分配客戶數: {len(customers)}")
        
        for customer in customers:
            inserted = False
            
            # 決定可用的車輛類型
            if customer.node_type == 'hard_to_access':
                # Hard-to-Access 只能由 UAV 服務
                candidate_routes = solution.uav_routes
                vehicle_type = 'uav'
            else:
                # 其他節點優先嘗試 MCS，再嘗試 UAV
                candidate_routes = solution.mcs_routes + solution.uav_routes
                vehicle_type = 'mcs'  # 預設開新車時使用 MCS
            
            # 嘗試插入現有路徑
            for route in candidate_routes:
                # 如果是 hard_to_access，跳過 MCS 路徑
                if customer.node_type == 'hard_to_access' and route.vehicle_type == 'mcs':
                    continue
                
                if self.try_insert_node(route, customer):
                    inserted = True
                    break
            
            # 如果無法插入現有路徑，開啟新車
            if not inserted:
                if customer.node_type == 'hard_to_access':
                    # 開啟新 UAV
                    new_route = Route(vehicle_type='uav')
                    new_route.add_node(customer)
                    if self.evaluate_route(new_route):
                        solution.add_uav_route(new_route)
                        inserted = True
                    else:
                        # 無法服務此節點
                        solution.unassigned_nodes.append(customer)
                else:
                    # 先嘗試開啟新 MCS
                    new_route = Route(vehicle_type='mcs')
                    new_route.add_node(customer)
                    if self.evaluate_route(new_route):
                        solution.add_mcs_route(new_route)
                        inserted = True
                    else:
                        # 嘗試開啟新 UAV
                        new_route = Route(vehicle_type='uav')
                        new_route.add_node(customer)
                        if self.evaluate_route(new_route):
                            solution.add_uav_route(new_route)
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
        hard_nodes = [n for n in self.nodes if n.node_type == 'hard_to_access']
        urgent_nodes = [n for n in self.nodes if n.node_type == 'urgent']
        
        # 繪製普通節點 (藍色)
        if normal_nodes:
            plt.scatter([n.x for n in normal_nodes], [n.y for n in normal_nodes],
                       c='blue', marker='o', s=50, label='Normal')
        
        # 繪製 Hard-to-Access 節點 (紅色)
        if hard_nodes:
            plt.scatter([n.x for n in hard_nodes], [n.y for n in hard_nodes],
                       c='red', marker='^', s=80, label='Hard-to-Access (10%)')
        
        # 繪製 Urgent 節點 (橘色)
        if urgent_nodes:
            plt.scatter([n.x for n in urgent_nodes], [n.y for n in urgent_nodes],
                       c='orange', marker='s', s=80, label='Urgent (20%)')
        
        # 繪製 Depot (綠色)
        if self.depot:
            plt.scatter(self.depot.x, self.depot.y,
                       c='green', marker='*', s=200, label='Depot')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('R101 Node Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"圖片已儲存至: {save_path}")
    
    def print_config(self) -> None:
        """輸出參數配置"""
        print("\n" + "="*50)
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
        print("="*50 + "\n")


# ==================== 主程式入口 ====================
def main():
    # 初始化問題
    problem = ChargingSchedulingProblem()
    
    # 輸出參數配置
    problem.print_config()
    
    # 載入資料
    problem.load_data('R101_25.csv')
    
    # 分配節點類型
    problem.assign_node_types()
    
    # 繪製節點分布圖
    problem.plot_nodes()
    
    # 輸出節點資訊
    print(f"\nHard-to-Access 節點索引: {sorted(problem.hard_to_access_indices)}")
    print(f"Urgent 節點索引: {sorted(problem.urgent_indices)}")
    
    # 執行 Greedy Construction Heuristic
    solution = problem.greedy_construction()
    
    # 輸出解的摘要
    solution.print_summary()


if __name__ == "__main__":
    main()

