import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Callable, Dict, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ==================== 參數配置 ====================
@dataclass
class MCSConfig:
    """Mobile Charging Station (MCS) 參數配置"""
    SPEED: float = (12 * 3.6) / 60.0    # km/min (原速度 12 m/s)
    CAPACITY: float = 270.0             # kWh 電池容量
    POWER_FAST: float = 250.0           # kW (Super Fast Charging)
    POWER_SLOW: float = 11.0            # kW (AC Slow Charging)


@dataclass(frozen=True)
class UAVConfig:
    """T-400 (engine-powered) UAV parameters for routing + mobile fast-charge module"""

    # --- Flight characteristics (NOT battery-driven) ---
    SPEED_KM_PER_MIN: float = 60.0 / 60.0      # cruise speed (km/min)
    ENDURANCE_MIN: float = 360.0               # 6 hours
    MAX_RANGE_KM: float = 200.0                # operational range / radius constraint

    # --- Charging module carried by UAV (this is the kWh you can deliver) ---
    CHARGE_POWER_KW: float = 50.0              # fast-charge output power
    DELIVERABLE_ENERGY_KWH: float = 20.0       # recommended: 10~20 kWh (payload-feasible)

    # --- Optional: operations overhead (helps realism in scheduling) ---
    TAKEOFF_LAND_OVERHEAD_MIN: float = 3.0     # handling/landing/safety buffer


@dataclass
class ProblemConfig:
    """問題實例參數配置"""
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
        
        Objective Function (Hierarchical):
            Cost = α × 未服務節點數 + β × 平均用戶等待時間 + γ × 總距離
        
        用戶等待時間定義 (Waiting for Service Arrival):
            - 用戶在 READY_TIME 發出充電請求
            - 用戶等待時間 = MCS 到達時間 - READY_TIME = arrival_time - ready_time
            - 這是系統的「響應時間」(Response Time)
            - 不包含充電服務時間
        
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
        
        # 2. 計算用戶等待時間 (用戶發出請求後等待 MCS 到達的時間 = arrival_time - ready_time)
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
        """
        讀取 CSV 資料並進行輸入驗證
        
        Validates:
        - Required columns exist
        - No NaN values in required numeric fields
        - NODE_TYPE values are valid ('depot', 'normal', 'urgent')
        
        Args:
            filepath: CSV 檔案路徑
            
        Raises:
            ValueError: 當 CSV 資料驗證失敗時
        """
        df = pd.read_csv(filepath)
        
        # ===== 1. 驗證必要欄位是否存在 =====
        required_columns = ['CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 
                           'READY TIME', 'DUE DATE', 'SERVICE TIME']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"CSV 缺少必要欄位: {missing_columns}\n"
                f"必要欄位: {required_columns}"
            )
        
        # ===== 2. 驗證數值欄位是否有 NaN =====
        numeric_columns = ['CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 
                          'READY TIME', 'DUE DATE', 'SERVICE TIME']
        nan_report = []
        for col in numeric_columns:
            nan_rows = df[df[col].isna()].index.tolist()
            if nan_rows:
                nan_report.append(f"  - {col}: 第 {[r + 2 for r in nan_rows]} 行 (CSV 行號)")
        
        if nan_report:
            raise ValueError(
                f"CSV 中發現 NaN 缺失值:\n" + "\n".join(nan_report)
            )
        
        # ===== 3. 驗證 NODE_TYPE 欄位 (若存在) =====
        has_node_type = 'NODE_TYPE' in df.columns
        valid_node_types = {'depot', 'normal', 'urgent'}
        
        if has_node_type:
            # 檢查 NODE_TYPE 是否有 NaN
            nan_node_type_rows = df[df['NODE_TYPE'].isna()].index.tolist()
            if nan_node_type_rows:
                raise ValueError(
                    f"NODE_TYPE 欄位存在 NaN 缺失值:\n"
                    f"  第 {[r + 2 for r in nan_node_type_rows]} 行 (CSV 行號)"
                )
            
            # 檢查 NODE_TYPE 值是否有效
            df['_node_type_lower'] = df['NODE_TYPE'].astype(str).str.strip().str.lower()
            invalid_mask = ~df['_node_type_lower'].isin(valid_node_types)
            invalid_rows = df[invalid_mask]
            
            if not invalid_rows.empty:
                invalid_report = []
                for idx, row in invalid_rows.iterrows():
                    invalid_report.append(
                        f"  - 第 {idx + 2} 行: NODE_TYPE='{row['NODE_TYPE']}'"
                    )
                raise ValueError(
                    f"CSV 中發現無效的 NODE_TYPE 值:\n" + "\n".join(invalid_report) +
                    f"\n有效的 NODE_TYPE 值: {valid_node_types}"
                )
            
            # 清理臨時欄位
            df = df.drop(columns=['_node_type_lower'])
        
        for idx, row in df.iterrows():
            # 讀取節點類型 (若有)
            if has_node_type:
                node_type = str(row['NODE_TYPE']).strip().lower()
            else:
                node_type = 'normal'
            
            ready_time = float(row['READY TIME'])
            due_date = float(row['DUE DATE'])
            
            node = Node(
                id=int(row['CUST NO.']),
                x=float(row['XCOORD.']),
                y=float(row['YCOORD.']),
                demand=float(row['DEMAND']),
                ready_time=ready_time,
                due_date=due_date,
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
            return distance / self.uav.SPEED_KM_PER_MIN
    
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
        capacity = self.mcs.CAPACITY if vehicle == 'mcs' else self.uav.DELIVERABLE_ENERGY_KWH
        
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
            # ===== 動態派遣約束：不能在 READY_TIME 之前出發 =====
            # 車輛必須等到客戶發出請求後才能開始前往該客戶
            # 這代表真實的動態調度情境：我們無法預知客戶何時會發出請求
            earliest_departure = node.ready_time
            actual_departure_from_prev = max(current_time, earliest_departure)
            
            # 計算行駛距離與時間
            travel_time = self.calculate_travel_time(prev_node, node, vehicle)
            if vehicle == 'mcs':
                distance = self.calculate_distance(prev_node, node, 'manhattan')
            else:
                distance = self.calculate_distance(prev_node, node, 'euclidean')
            
            route.total_distance += distance
            arrival_time = actual_departure_from_prev + travel_time
            
            # 檢查是否能在 due_date 前到達
            if arrival_time > node.due_date:
                route.is_feasible = False
                return False
            
            # ===== 等待時間計算 =====
            # 由於不能提前出發，mcs_waiting_time 理論上為 0
            # (車輛在前一個位置等待，而非在客戶位置等待)
            mcs_waiting_time = 0.0
            
            # 到達即開始服務 (因為到達時間已經 >= ready_time)
            service_start = arrival_time
            
            # ===== 動態充電模式選擇 =====
            if vehicle == 'uav':
                # UAV: 使用充電模組功率
                charging_power = self.uav.CHARGE_POWER_KW
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
            
            # ===== 計算用戶等待時間 (純等待到達) =====
            # user_waiting_time = 用戶從發出請求到 MCS 到達的時間
            user_waiting_time = arrival_time - node.ready_time
            
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
        
        # 檢查能量/範圍約束
        # MCS: 假設每公里消耗 0.5 kWh
        # UAV (T-400): 引擎動力，檢查作業範圍約束
        if vehicle == 'mcs':
            energy_consumed = route.total_distance * 0.5
            if energy_consumed > self.mcs.CAPACITY:
                route.is_feasible = False
                return False
        else:
            # UAV 使用作業範圍約束而非電池容量
            if route.total_distance > self.uav.MAX_RANGE_KM * 2:  # 來回距離
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
            capacity = self.uav.DELIVERABLE_ENERGY_KWH
        
        if route.total_demand + node.demand > capacity:
            return False, float('inf')
        
        # ===== 2. 計算到達新節點的時間 O(1) =====
        # 動態派遣約束：不能在 READY_TIME 之前出發
        if pos == 0:
            # 從 depot 出發，但必須等到 node.ready_time
            prev_departure_time = 0.0
            prev_node = self.depot
        else:
            prev_departure_time = route.departure_times[pos - 1]
            prev_node = route.nodes[pos - 1]
        
        # 不能在客戶發出請求前出發
        earliest_departure = node.ready_time
        actual_departure = max(prev_departure_time, earliest_departure)
        
        travel_time_to_new = self.calculate_travel_time(prev_node, node, vehicle)
        arrival_at_new = actual_departure + travel_time_to_new
        
        # 檢查是否能在 due_date 前到達
        if arrival_at_new > node.due_date:
            return False, float('inf')
        
        # ===== 3. 決定充電模式並計算服務時間 O(1) =====
        # 到達即開始服務 (因為到達時間已經 >= ready_time)
        service_start_new = arrival_at_new
        
        if vehicle == 'uav':
            charging_power = self.uav.CHARGE_POWER_KW
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
        new_node_waiting = arrival_at_new - node.ready_time
        
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
        
        # ===== 5.5 Check Energy/Range Constraints (O(1)) =====
        # Calculate change in distance
        delta_dist = 0.0
        dist_type = 'manhattan' if vehicle == 'mcs' else 'euclidean'
        
        # prev_node is already defined above at step 2
        dist_prev_new = self.calculate_distance(prev_node, node, dist_type)
        delta_dist += dist_prev_new
        
        if pos < len(route.nodes):
            next_node = route.nodes[pos]
            dist_new_next = self.calculate_distance(node, next_node, dist_type)
            dist_prev_next = self.calculate_distance(prev_node, next_node, dist_type)
            delta_dist += (dist_new_next - dist_prev_next)
        else:
            # End of route: prev -> node -> depot vs prev -> depot
            dist_node_return = self.calculate_distance(node, self.depot, dist_type)
            dist_prev_return = self.calculate_distance(prev_node, self.depot, dist_type)
            delta_dist += (dist_node_return - dist_prev_return)
            
        new_total_distance = route.total_distance + delta_dist
        
        if vehicle == 'mcs':
             # MCS Energy Check (0.5 kWh/km)
             energy_consumed = new_total_distance * 0.5 
             if energy_consumed > self.mcs.CAPACITY:
                  return False, float('inf')
        else:
             # UAV Range Check
             if new_total_distance > self.uav.MAX_RANGE_KM * 2:
                  return False, float('inf')

        # ===== 6. 計算總 delta cost =====
        delta_cost = new_node_waiting + delta_successor_waiting
        
        return True, delta_cost
    
    def _forward_propagate_check(self, route: Route, start_pos: int, 
                                  time_shift: float) -> Tuple[bool, float]:
        """
        前向傳播時間偏移檢查 (考慮動態派遣約束)
        
        檢查從 start_pos 開始，若到達時間延後 time_shift，後續節點是否仍可行。
        同時計算用戶等待時間的變化量。
        
        動態派遣約束：不能在客戶 READY_TIME 之前出發前往該客戶
        
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
            
            # 動態派遣約束下，到達即開始服務 (arrival >= ready_time)
            old_service_start = old_arrival  # 舊邏輯下也是到達即服務
            new_service_start = new_arrival
            
            # 充電時間不變 (demand 和 power 不變)
            charging_time = route.departure_times[i] - old_service_start
            
            new_departure = new_service_start + charging_time
            
            # 檢查服務是否能在 due_date 前完成
            if new_departure > node.due_date:
                return False, float('inf')
            
            # 計算等待時間變化 (Waiting Time = Arrival - Ready)
            old_waiting = route.user_waiting_times[i]
            new_waiting = new_arrival - node.ready_time
            delta_waiting += (new_waiting - old_waiting)
            
            # 更新時間偏移量
            # 動態派遣約束：下一個節點的出發時間 = max(new_departure, next_node.ready_time)
            # 所以偏移量可能因為 ready_time 約束而被「重置」
            old_departure = route.departure_times[i]
            
            if i + 1 < len(route.nodes):
                next_node = route.nodes[i + 1]
                # 新的出發時間 (考慮動態約束)
                new_depart_to_next = max(new_departure, next_node.ready_time)
                old_depart_to_next = max(old_departure, next_node.ready_time)
                current_shift = new_depart_to_next - old_depart_to_next
            else:
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

    
    def cluster_customers(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        對客戶進行時空分群 (Spatial + Temporal Clustering)
        
        Returns:
            labels: 每個節點的群聚標籤
            centroids: 群聚中心座標 (K, 2)
            n_clusters: 群聚數量
        """
        # 1. 準備特徵數據 (X, Y, ReadyTime)
        customers = [n for n in self.nodes if n.node_type != 'depot']
        features = []
        
        for n in customers:
            features.append([n.x, n.y, n.ready_time])
            
        data = np.array(features)
        
        # 2. 標準化 (因為時間和空間的尺度不同)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # 3. 決定最佳群數 (使用 Silhouette Score)
        best_score = -1
        best_k = 2
        
        # 嘗試 k 從 3 到 sqrt(N) * 2 或 15 (視規模而定)
        min_k = 3
        max_k = min(len(customers) // 2, 15)  # 避免過多群
        if max_k < min_k: 
            max_k = min_k
            
        # 如果客戶太少，直接視為一群或兩群
        if len(customers) < min_k + 2:
            kmeans = KMeans(n_clusters=1, random_state=self.config.RANDOM_SEED, n_init=10)
            labels = kmeans.fit_predict(data_scaled)
            return labels, kmeans.cluster_centers_[:, :2], 1  # Centroids 只取前兩維 (x, y) ? 
            # 注意: KMeans centroid 是在 scaled space，需要 inverse transform 才能得到真實座標
            # 這裡簡化：MCS 預派遣位置可以設在該群 "所有節點座標的平均值" (真實空間)
        
        best_labels = None
        
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.config.RANDOM_SEED, n_init=10)
            labels = kmeans.fit_predict(data_scaled)
            
            # 只有一類無法計算 silhouette
            if len(set(labels)) < 2:
                continue
                
            score = silhouette_score(data_scaled, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        
        # 4. 根據最佳 k 重新訓練或直接使用
        n_clusters = best_k
        labels = best_labels
        
        # 5. 計算真實空間的群聚中心 (Centroids)
        # 用於 MCS 預派遣位置
        centroids = []
        for k in range(n_clusters):
            # 找出屬於該群的客戶
            cluster_indices = [i for i, label in enumerate(labels) if label == k]
            cluster_nodes = [customers[i] for i in cluster_indices]
            
            if not cluster_nodes:
                continue
                
            # 平均位置
            avg_x = np.mean([n.x for n in cluster_nodes])
            avg_y = np.mean([n.y for n in cluster_nodes])
            centroids.append((avg_x, avg_y))
            
        return labels, np.array(centroids), n_clusters

    def cluster_based_construction(self) -> Solution:
        """
        Cluster-Based Construction Heuristic
        
        策略：
        1. 使用 K-Means 對客戶進行分群 (考量位置 + Ready Time)
        2. 每個群指派一台 MCS，位置設為群中心 (Centroid)
        3. 該 MCS 優先服務群內客戶
        4. 群內客戶排序：Earliest Due Date First (EDD)
        5. 當群內客戶無法被 MCS 服務時 (時間窗/電量)，嘗試 UAV
        
        Returns:
            初始解
        """
        solution = Solution()
        customers = [n for n in self.nodes if n.node_type != 'depot']
        
        # Step 1: 分群
        labels, centroids, n_clusters = self.cluster_customers()
        
        print(f"K-Means Clustering Result: k={n_clusters}")
        
    def parallel_insertion_construction(self) -> Solution:
        """
        Parallel Insertion Construction Heuristic
        
        策略：
        1. 全局排序：優先處理 Urgent 節點，其次依 Due Date (EDD)
        2. 平行插入：嘗試將每個客戶插入現有的 MCS 路徑中 (最小化邊際成本)
        3. 動態開車：若無法插入現有路徑，則開啟新 MCS 路徑
        4. UAV 補救：若 MCS 無法服務 (時間窗/電量)，嘗試 UAV
        
        這種 "Parallel Insertion" 策略比 Cluster-First 更靈活，
        特別適合處理 Urgent 節點和緊迫的時間窗。
        
        Returns:
            初始解
        """
        solution = Solution()
        customers = [n for n in self.nodes if n.node_type != 'depot']
        
        # Step 1: 排序 (Sorting)
        # 排序權重：Urgent (-1) 優先於 Normal (0)，同類型依 Due Date 排序
        # 這樣可以確保 Urgent 節點優先被安排
        customers.sort(key=lambda n: (0 if n.node_type == 'urgent' else 1, n.due_date))
        
        print(f"Sorted Customers ({len(customers)}):")
        for n in customers:
            print(f"  - Node {n.id}: Type={n.node_type}, Ready={n.ready_time}, Due={n.due_date:.1f}, Demand={n.demand}")
        
        unassigned = []
        
        # Step 2: 平行插入 (Parallel Insertion)
        for node in customers:
            best_route_idx = -1
            best_position = -1
            min_marginal_cost = float('inf')
            
            # 2.1 嘗試插入現有 MCS 路徑
            for r_idx, route in enumerate(solution.mcs_routes):
                # 嘗試所有位置
                for pos in range(len(route.nodes) + 1):
                    # 使用增量檢查 O(1)
                    feasible, delta_cost = self.incremental_insertion_check(route, pos, node)
                    
                    if feasible and delta_cost < min_marginal_cost:
                        min_marginal_cost = delta_cost
                        best_route_idx = r_idx
                        best_position = pos
            
            # 2.2 執行最佳插入
            inserted_successfully = False
            if best_route_idx != -1:
                target_route = solution.mcs_routes[best_route_idx]
                target_route.insert_node(best_position, node)
                # 更新路徑狀態 (因為 incremental_check 不會修改路徑)
                if self.evaluate_route(target_route):
                    inserted_successfully = True
                else:
                    # [Safety] 若插入後導致路徑不可行 (例如 incremental check 漏掉某些約束)，則還原
                    target_route.remove_node(best_position)
                    self.evaluate_route(target_route) # Restore state
                    print(f"Warning: Reverted insertion of Node {node.id} to Route {target_route.vehicle_id} (Infeasible after check)")
            
            if not inserted_successfully:
                # 2.3 若無法插入現有路徑，嘗試開啟新 MCS 路徑
                new_route = Route(vehicle_type='mcs', vehicle_id=len(solution.mcs_routes))
                new_route.add_node(node)
                
                if self.evaluate_route(new_route):
                    solution.add_mcs_route(new_route)
                else:
                    # 連新開 MCS 都無法服務 (可能太遠或時間窗太緊)
                    unassigned.append(node)
        
        # Step 3: 嘗試用 UAV 服務未分配節點 (MCS 遺漏的)
        final_unassigned = []
        
        for node in unassigned:
            served_by_uav = False
            
            # 先嘗試插入現有 UAV 路徑
            for uav_route in solution.uav_routes:
                if self.try_insert_node(uav_route, node):
                    served_by_uav = True
                    break
            
            # 若現有 UAV 無法服務，開新 UAV
            if not served_by_uav:
                new_uav_route = Route(vehicle_type='uav', vehicle_id=len(solution.uav_routes))
                if self.try_insert_node(new_uav_route, node):
                    solution.add_uav_route(new_uav_route)
                    served_by_uav = True
            
            if not served_by_uav:
                final_unassigned.append(node)
                
        solution.unassigned_nodes = final_unassigned
        solution.calculate_total_cost(len(customers))
        
        return solution

    def plot_nodes(self, save_path: str = 'node_distribution.png', centroids: np.ndarray = None, cluster_radii: List[float] = None) -> None:
        """繪製節點分布圖 (包含 Cluster Centroids 與邊界)"""
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
        
        # 繪製 Cluster Centroids (若有 provided)
        if centroids is not None and len(centroids) > 0:
            plt.scatter(centroids[:, 0], centroids[:, 1],
                       c='red', marker='x', s=100, linewidths=2, label='Cluster Centroids')
            
            # 繪製群聚範圍 (紅色虛線圓形)
            if cluster_radii:
                ax = plt.gca()
                for i, (cx, cy) in enumerate(centroids):
                    if i < len(cluster_radii):
                        radius = cluster_radii[i]
                        # 增加一點緩衝讓圓圈不要切到點
                        circle = plt.Circle((cx, cy), radius * 1.1, color='red', fill=False, linestyle='--', alpha=0.5, linewidth=1.5)
                        ax.add_patch(circle)
        
        # 繪製 Depot (綠色)
        if self.depot:
            plt.scatter(self.depot.x, self.depot.y,
                       c='green', marker='*', s=200, label='Depot')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('R101 Node Distribution')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1, frameon=True) # 調整 Legend 位置
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
        print("\nUAV (T-400 Engine-Powered) 參數配置:")
        print(f"  巡航速度: {self.uav.SPEED_KM_PER_MIN:.3f} km/min ({self.uav.SPEED_KM_PER_MIN * 60:.1f} km/h)")
        print(f"  續航時間: {self.uav.ENDURANCE_MIN} min ({self.uav.ENDURANCE_MIN / 60:.1f} hours)")
        print(f"  最大作業範圍: {self.uav.MAX_RANGE_KM} km")
        print(f"  充電模組輸出功率: {self.uav.CHARGE_POWER_KW} kW")
        print(f"  可傳輸電量: {self.uav.DELIVERABLE_ENERGY_KWH} kWh")
        print(f"  起降作業開銷: {self.uav.TAKEOFF_LAND_OVERHEAD_MIN} min")
        print("="*80 + "\n")


# ==================== 主程式入口 ====================
def main():
    # 初始化問題
    problem = ChargingSchedulingProblem()
    
    # 輸出參數配置
    problem.print_config()
    
    # 載入資料 (使用 mixed_instance.csv，節點類型已在 CSV 中定義)
    problem.load_data('instance_25c_random_s42.csv')
    
    # 分配節點類型 (使用 CSV 中的類型定義)
    problem.assign_node_types(use_csv_types=True)
    
    # 計算群聚 (僅用於視覺化節點分布)
    print("Calculating clusters for visualization...")
    labels, plot_centroids, n_clusters = problem.cluster_customers()
    
    # 計算每個群聚的半徑 (最大距離)
    cluster_radii = []
    customers = [n for n in problem.nodes if n.node_type != 'depot']
    
    for k in range(n_clusters):
        # 找出屬於該群的索引
        indices = [i for i, label in enumerate(labels) if label == k]
        if not indices:
            cluster_radii.append(0.0)
            continue
        
        # 取得該群所有點的座標
        points = np.array([[customers[i].x, customers[i].y] for i in indices])
        centroid = plot_centroids[k]
        
        # 計算每個點到中心點的歐幾里得距離
        dists = np.sqrt(np.sum((points - centroid)**2, axis=1))
        
        # 取最大距離作為半徑
        max_dist = np.max(dists)
        cluster_radii.append(max_dist)
    
    # 繪製節點分布圖
    problem.plot_nodes(centroids=plot_centroids, cluster_radii=cluster_radii)
    
    # ==================== Parallel Insertion Construction ====================
    print("\n" + "="*80)
    print("Phase 2: Parallel Insertion Construction (Sorted by Necessity)")
    print("="*80)
    
    start_time = time.time()
    # 改用 Parallel Insertion Strategy
    solution = problem.parallel_insertion_construction()
    elapsed_time = time.time() - start_time
    
    solution.print_summary()
    print(f"\nExecution Time: {elapsed_time:.4f} seconds")
    
    # 繪製解路徑
    plot_routes(solution, problem, save_path='routes_insertion.png')
    
    print("-" * 72)
    print(f"\n圖片已儲存:")
    print(f"  - 節點分布圖: node_distribution.png")
    print(f"  - 路線圖: routes_insertion.png")


if __name__ == "__main__":
    main()
