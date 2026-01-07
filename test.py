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
        print(f"平均等待時間: {self.avg_waiting_time:.2f} 分鐘 ← 主要目標")
        print(f"覆蓋率: {self.coverage_rate:.1%} ({self.total_customers - len(self.unassigned_nodes)}/{self.total_customers})")
        print(f"可行解: {'是' if self.is_feasible else '否 (有未服務節點)'}")
        
        # 其他指標
        print("\n" + "其他指標")
        print(f"總距離: {self.total_distance:.2f} km")
        print(f"總時間: {self.total_time:.2f} 分鐘")
        print(f"總等待時間: {self.total_waiting_time:.2f} 分鐘")
        print(f"彈性分數 (CV): {self.flexibility_score:.3f}")
        
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
                      f"等待={route.total_waiting_time:.1f}min, "
                      f"模式={mode_summary}")
        
        if self.uav_routes:
            print("\n" + "UAV 路徑詳情")
            for route in self.uav_routes:
                print(f"{route.vehicle_type.upper()}-{route.vehicle_id + 1}: "
                      f"節點={route.get_node_ids()}, "
                      f"距離={route.total_distance:.1f}km, "
                      f"等待={route.total_waiting_time:.1f}min")
        
        print("\n" + "="*60)