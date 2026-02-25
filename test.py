class ALNSSolver:
    """
    Adaptive Large Neighborhood Search (ALNS) 求解器
    
    結合 Simulated Annealing 退火機制與 Roulette Wheel 算子選擇，
    透過 Destroy–Repair 迭代搜尋改善初始解。
    """
    
    def __init__(self, problem: ChargingSchedulingProblem, config: ALNSConfig = None):
        self.problem = problem
        self.cfg = config or ALNSConfig()
        
        # ===== 算子登錄 =====
        self.destroy_ops: Dict[str, Callable] = {
            'random_removal': self._random_removal,
            'worst_removal':  self._worst_removal,
        }
        self.repair_ops: Dict[str, Callable] = {
            'greedy_insertion': self._greedy_insertion,
            'regret2_insertion': self._regret2_insertion,
        }
        
        # ===== 算子權重 (初始化為均等) =====
        self.destroy_weights: Dict[str, float] = {k: 1.0 for k in self.destroy_ops}
        self.repair_weights:  Dict[str, float] = {k: 1.0 for k in self.repair_ops}
        
        # ===== Segment 統計 =====
        self._reset_segment_stats()
    
    # ------------------------------------------------------------------
    #  外層迴圈
    # ------------------------------------------------------------------
    def solve(self, initial_solution: Solution) -> Solution:
        """
        ALNS 主迴圈
        
        Args:
            initial_solution: 由建構式啟發式產生的初始解
        
        Returns:
            全局最佳解 (best_solution)
        """
        current = initial_solution.copy()
        best = initial_solution.copy()
        temperature = self.cfg.SA_INITIAL_TEMP
        
        total_customers = current.total_customers
        
        # 計算摧毀節點數的範圍
        served_count = sum(len(r.nodes) for r in current.mcs_routes)
        min_remove = max(1, int(served_count * self.cfg.DESTROY_RATIO_MIN))
        max_remove = max(min_remove + 1, int(served_count * self.cfg.DESTROY_RATIO_MAX))
        
        print(f"\n{'='*60}")
        print(f"ALNS 開始 | 初始 cost={best.total_cost:.4f} | "
              f"destroy range=[{min_remove}, {max_remove}]")
        print(f"{'='*60}")
        
        for iteration in range(1, self.cfg.MAX_ITERATIONS + 1):
            # ----- 1. 選算子 (Roulette Wheel) -----
            d_name = self._roulette_select(self.destroy_weights)
            r_name = self._roulette_select(self.repair_weights)
            
            # ----- 2. Destroy -----
            candidate = current.copy()
            num_remove = random.randint(min_remove, max_remove)
            removed_nodes = self.destroy_ops[d_name](candidate, num_remove)
            
            # ----- 3. Repair -----
            self.repair_ops[r_name](candidate, removed_nodes)
            candidate.calculate_total_cost(total_customers)
            
            # ----- 4. SA Acceptance -----
            delta = candidate.total_cost - current.total_cost
            accepted = False
            
            if delta < -1e-9:
                # 改善解，直接接受
                current = candidate
                accepted = True
                
                if current.total_cost < best.total_cost - 1e-9:
                    # 🏆 新全局最佳
                    best = current.copy()
                    self._record_score(d_name, r_name, self.cfg.SIGMA_1)
                else:
                    self._record_score(d_name, r_name, self.cfg.SIGMA_2)
            elif self._sa_accept(delta, temperature):
                # SA 接受較差解
                current = candidate
                accepted = True
                self._record_score(d_name, r_name, self.cfg.SIGMA_3)
            # else: 拒絕，不記分
            
            # ----- 5. 冷卻 -----
            temperature = max(temperature * self.cfg.SA_COOLING_RATE, self.cfg.SA_FINAL_TEMP)
            
            # ----- 6. Segment 結束 → 更新權重 -----
            if iteration % self.cfg.SEGMENT_SIZE == 0:
                self._update_weights()
                self._reset_segment_stats()
            
            # ----- 7. 日誌 -----
            if iteration % 500 == 0 or iteration == 1:
                print(f"  iter={iteration:5d} | best={best.total_cost:.4f} | "
                      f"current={current.total_cost:.4f} | T={temperature:.4f} | "
                      f"d={d_name} r={r_name}")
        
        print(f"\nALNS 結束 | best cost={best.total_cost:.4f} | "
              f"改善 {((initial_solution.total_cost - best.total_cost) / initial_solution.total_cost * 100):.2f}%")
        
        # 印出最終算子權重
        print("\n最終算子權重:")
        print(f"  Destroy: {self.destroy_weights}")
        print(f"  Repair:  {self.repair_weights}")
        
        return best
    
    # ------------------------------------------------------------------
    #  Roulette Wheel Selection
    # ------------------------------------------------------------------
    def _roulette_select(self, weights: Dict[str, float]) -> str:
        """輪盤賭選擇算子"""
        names = list(weights.keys())
        vals = [weights[n] for n in names]
        total = sum(vals)
        
        r = random.random() * total
        cumulative = 0.0
        for name, val in zip(names, vals):
            cumulative += val
            if r <= cumulative:
                return name
        return names[-1]  # fallback
    
    # ------------------------------------------------------------------
    #  Simulated Annealing Acceptance
    # ------------------------------------------------------------------
    def _sa_accept(self, delta: float, temperature: float) -> bool:
        """SA 接受準則: delta >= 0 時以 exp(-delta/T) 機率接受"""
        if temperature < 1e-12:
            return False
        prob = np.exp(-delta / temperature)
        return random.random() < prob
    
    # ------------------------------------------------------------------
    #  Segment 統計 & 權重更新
    # ------------------------------------------------------------------
    def _reset_segment_stats(self):
        """重置 segment 統計"""
        self._destroy_scores: Dict[str, float] = {k: 0.0 for k in self.destroy_ops}
        self._repair_scores:  Dict[str, float] = {k: 0.0 for k in self.repair_ops}
        self._destroy_usage:  Dict[str, int]   = {k: 0   for k in self.destroy_ops}
        self._repair_usage:   Dict[str, int]   = {k: 0   for k in self.repair_ops}
    
    def _record_score(self, d_name: str, r_name: str, score: float):
        """記錄算子得分"""
        self._destroy_scores[d_name] += score
        self._repair_scores[r_name]  += score
        self._destroy_usage[d_name]  += 1
        self._repair_usage[r_name]   += 1
    
    def _update_weights(self):
        """
        Segment 結束時更新算子權重
        
        w_new = (1 - λ) × w_old + λ × (score / usage)
        """
        lam = self.cfg.REACTION_FACTOR
        
        for name in self.destroy_weights:
            usage = self._destroy_usage[name]
            if usage > 0:
                avg_score = self._destroy_scores[name] / usage
                self.destroy_weights[name] = (1 - lam) * self.destroy_weights[name] + lam * avg_score
            # 下界保護 (避免權重趨近於 0 而永遠不被選到)
            self.destroy_weights[name] = max(self.destroy_weights[name], 0.1)
        
        for name in self.repair_weights:
            usage = self._repair_usage[name]
            if usage > 0:
                avg_score = self._repair_scores[name] / usage
                self.repair_weights[name] = (1 - lam) * self.repair_weights[name] + lam * avg_score
            self.repair_weights[name] = max(self.repair_weights[name], 0.1)
    
    # ==================================================================
    #  Destroy Operators
    # ==================================================================
    def _random_removal(self, solution: Solution, num_remove: int) -> List[Node]:
        """Random Removal: 隨機從所有路徑中移除節點 (修復 UAV 盲點)"""
        # 收集所有可移除的 (車型, 路線索引, 節點位置) 對
        candidates: List[Tuple[str, int, int]] = []
        
        for r_idx, route in enumerate(solution.mcs_routes):
            for n_pos in range(len(route.nodes)):
                candidates.append(('mcs', r_idx, n_pos))
                
        for r_idx, route in enumerate(solution.uav_routes):
            for n_pos in range(len(route.nodes)):
                candidates.append(('uav', r_idx, n_pos))
        
        if not candidates:
            return []
        
        num_remove = min(num_remove, len(candidates))
        selected = random.sample(candidates, num_remove)
        
        # 🚨 關鍵：為了避免索引偏移，必須按照位置 (n_pos) 由大到小排序，從後面往前拔！
        selected.sort(key=lambda x: x[2], reverse=True)
        
        removed_nodes: List[Node] = []
        for v_type, r_idx, n_pos in selected:
            if v_type == 'mcs':
                node = solution.mcs_routes[r_idx].remove_node(n_pos)
            else:
                node = solution.uav_routes[r_idx].remove_node(n_pos)
            removed_nodes.append(node)
        
        self._cleanup_routes(solution)
        return removed_nodes
    
    def _worst_removal(self, solution: Solution, num_remove: int) -> List[Node]:
        """
        Worst Removal: 移除「成本貢獻最高」的節點 (包含 MCS 與 UAV)
        """
        # 1. 收集所有節點的成本貢獻 (cost, vehicle_type, route_idx, node_pos)
        node_costs: List[Tuple[float, str, int, int]] = []
        
        for v_type, routes in [('mcs', solution.mcs_routes), ('uav', solution.uav_routes)]:
            for r_idx, route in enumerate(routes):
                for n_pos, node in enumerate(route.nodes):
                    if n_pos < len(route.user_waiting_times):
                        cost = route.user_waiting_times[n_pos]
                    else:
                        cost = 0.0
                    node_costs.append((cost, v_type, r_idx, n_pos))
        
        if not node_costs:
            return []
            
        # 2. 按成本降序排列 (最糟的在最前面)
        node_costs.sort(key=lambda x: x[0], reverse=True)
        
        # 3. 挑出要移除的候選人
        selected_to_remove = []
        num_remove = min(num_remove, len(node_costs))
        
        for _ in range(num_remove):
            if not node_costs:
                break
            # Shaw's stochastic selection
            rand_val = random.random()
            idx = int(rand_val ** self.cfg.WORST_REMOVAL_P * len(node_costs))
            idx = min(idx, len(node_costs) - 1)
            selected_to_remove.append(node_costs.pop(idx))
            
        # 🚨 4. 關鍵防呆：從後往前拔，避免 Index Shift 導致拔錯節點！
        # 排序條件：先按 vehicle_type, 再按 route_idx, 最後按 n_pos (降序)
        selected_to_remove.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        
        removed_nodes: List[Node] = []
        for cost, v_type, r_idx, n_pos in selected_to_remove:
            if v_type == 'mcs':
                node = solution.mcs_routes[r_idx].remove_node(n_pos)
            else:
                node = solution.uav_routes[r_idx].remove_node(n_pos)
            removed_nodes.append(node)
            
        self._cleanup_routes(solution)
        return removed_nodes
    
    def _cleanup_routes(self, solution: Solution):
        """刷新所有路徑狀態並移除空路徑"""
        # 重新評估所有 MCS 路徑
        for route in solution.mcs_routes:
            self.problem.evaluate_route(route)
        
        # 移除空路徑
        solution.mcs_routes = [r for r in solution.mcs_routes if len(r.nodes) > 0]
        
        # 重新編號
        for i, route in enumerate(solution.mcs_routes):
            route.vehicle_id = i

        for route in solution.uav_routes:
            self.problem.evaluate_route(route)
        solution.uav_routes = [r for r in solution.uav_routes if len(r.nodes) > 0]
        for i, route in enumerate(solution.uav_routes):
            route.vehicle_id = i
    
    # ==================================================================
    #  Repair Operators
    # ==================================================================
    def _greedy_insertion(self, solution: Solution, removed_nodes: List[Node]):
        """
        Greedy Insertion: 按 EDD 排序後逐一找全局最佳位置插入
        
        封裝自 parallel_insertion_construction 的插入邏輯，
        使用 incremental_insertion_check 評估。
        """
        # 按 due_date (EDD) 排序，Urgent 優先
        pool = removed_nodes + solution.unassigned_nodes
        pool.sort(key=lambda n: (0 if n.node_type == 'urgent' else 1, n.due_date))
        still_unassigned: List[Node] = []
        
        for node in pool:
            best_route_type: Optional[str] = None  # 'mcs' or 'uav' or None
            best_route_idx = -1
            best_position = -1
            min_cost = float('inf')
            
            # 嘗試插入現有 MCS 路徑
            for r_idx, route in enumerate(solution.mcs_routes):
                for pos in range(len(route.nodes) + 1):
                    feasible, delta_cost = self.problem.incremental_insertion_check(route, pos, node)
                    if feasible and delta_cost < min_cost:
                        min_cost = delta_cost
                        best_route_type = 'mcs'
                        best_route_idx = r_idx
                        best_position = pos
            
            # 嘗試插入現有 UAV 路徑 (僅限 Urgent)
            if node.node_type == 'urgent':
                for r_idx, route in enumerate(solution.uav_routes):
                    for pos in range(len(route.nodes) + 1):
                        feasible, delta_cost = self.problem.incremental_insertion_check(route, pos, node)
                        if feasible and delta_cost < min_cost:
                            min_cost = delta_cost
                            best_route_type = 'uav'
                            best_route_idx = r_idx
                            best_position = pos
            
            # 執行插入或動態開車
            if best_route_type == 'mcs':
                route = solution.mcs_routes[best_route_idx]
                route.insert_node(best_position, node)
                if not self.problem.evaluate_route(route):
                    route.remove_node(best_position)
                    self.problem.evaluate_route(route)
                    still_unassigned.append(node)      # 宣告失敗，丟回未分配
                
            elif best_route_type == 'uav':
                route = solution.uav_routes[best_route_idx]
                route.insert_node(best_position, node)
                if not self.problem.evaluate_route(route):
                    route.remove_node(best_position)
                    self.problem.evaluate_route(route)
                    still_unassigned.append(node)
            else:
                # 無法插入現有路徑 → 動態開新車 (支援 UAV 救援)
                inserted = False
                if node.node_type == 'urgent':
                    # Urgent 節點：PK 新開 MCS 與新開 UAV
                    nearest_centroid = self.problem._find_nearest_centroid(node)
                    new_mcs = Route(vehicle_type='mcs', vehicle_id=len(solution.mcs_routes))
                    new_mcs.start_node = nearest_centroid
                    new_mcs.add_node(node)
                    mcs_feasible = self.problem.evaluate_route(new_mcs)
                    mcs_cost = new_mcs.total_user_waiting_time if mcs_feasible else float('inf')
                    
                    new_uav = Route(vehicle_type='uav', vehicle_id=len(solution.uav_routes))
                    new_uav.add_node(node)
                    uav_feasible = self.problem.evaluate_route(new_uav)
                    uav_cost = new_uav.total_user_waiting_time if uav_feasible else float('inf')
                    
                    if mcs_feasible or uav_feasible:
                        if uav_cost <= mcs_cost:
                            solution.add_uav_route(new_uav)
                        else:
                            solution.add_mcs_route(new_mcs)
                        inserted = True
                else:
                    # Normal 節點：只能開新 MCS
                    nearest_centroid = self.problem._find_nearest_centroid(node)
                    new_mcs = Route(vehicle_type='mcs', vehicle_id=len(solution.mcs_routes))
                    new_mcs.start_node = nearest_centroid
                    new_mcs.add_node(node)
                    if self.problem.evaluate_route(new_mcs):
                        solution.add_mcs_route(new_mcs)
                        inserted = True
                        
                if not inserted:
                    still_unassigned.append(node)
        
        solution.unassigned_nodes = still_unassigned
    
    def _regret2_insertion(self, solution: Solution, removed_nodes: List[Node]):
        """
        Regret-2 Insertion: 每輪選擇「regret 值最大」的節點優先插入
        
        regret = best2_cost - best1_cost
        直覺：若某節點只有一個好位置，不優先處理的話後面就插不進去了。
        """
        pool = removed_nodes + solution.unassigned_nodes
        still_unassigned: List[Node] = []
        
        while pool:
            best_node = None
            best_node_idx = -1
            best_regret = -float('inf')
            best_insert_type: Optional[str] = None  # 'mcs' or 'uav'
            best_insert_route_idx = -1
            best_insert_pos = -1
            best_insert_cost = float('inf')
            
            for n_idx, node in enumerate(pool):
                # 收集所有可行插入位置的 (cost, route_type, route_idx, position)
                insertion_options: List[Tuple[float, str, int, int]] = []
                
                # MCS 路徑
                for r_idx, route in enumerate(solution.mcs_routes):
                    for pos in range(len(route.nodes) + 1):
                        feasible, delta_cost = self.problem.incremental_insertion_check(route, pos, node)
                        if feasible:
                            insertion_options.append((delta_cost, 'mcs', r_idx, pos))
                
                # UAV 路徑 (僅 Urgent)
                if node.node_type == 'urgent':
                    for r_idx, route in enumerate(solution.uav_routes):
                        for pos in range(len(route.nodes) + 1):
                            feasible, delta_cost = self.problem.incremental_insertion_check(route, pos, node)
                            if feasible:
                                insertion_options.append((delta_cost, 'uav', r_idx, pos))
                
                if not insertion_options:
                    # 完全無法插入 → 等開新車
                    continue
                
                # 排序取 top-2
                insertion_options.sort(key=lambda x: x[0])
                best1_cost, best1_type, best1_r, best1_p = insertion_options[0]
                
                if len(insertion_options) >= 2:
                    best2_cost = insertion_options[1][0]
                else:
                    best2_cost = best1_cost + 1e6  # 只有一個位置 → regret 極大
                
                regret = best2_cost - best1_cost
                
                # 選 regret 最大者；tie-break: best1_cost 最小(最便宜的)
                if (regret > best_regret + 1e-9 or 
                    (abs(regret - best_regret) < 1e-9 and best1_cost < best_insert_cost - 1e-9)):
                    best_regret = regret
                    best_node = node
                    best_node_idx = n_idx
                    best_insert_type = best1_type
                    best_insert_route_idx = best1_r
                    best_insert_pos = best1_p
                    best_insert_cost = best1_cost
            
            if best_node is None:
                # 剩餘節點都無法插入現有路徑 → 嘗試動態開新車 (支援 UAV 救援)
                for node in pool:
                    inserted = False
                    if node.node_type == 'urgent':
                        nearest_centroid = self.problem._find_nearest_centroid(node)
                        new_mcs = Route(vehicle_type='mcs', vehicle_id=len(solution.mcs_routes))
                        new_mcs.start_node = nearest_centroid
                        new_mcs.add_node(node)
                        mcs_feasible = self.problem.evaluate_route(new_mcs)
                        mcs_cost = new_mcs.total_user_waiting_time if mcs_feasible else float('inf')
                        
                        new_uav = Route(vehicle_type='uav', vehicle_id=len(solution.uav_routes))
                        new_uav.add_node(node)
                        uav_feasible = self.problem.evaluate_route(new_uav)
                        uav_cost = new_uav.total_user_waiting_time if uav_feasible else float('inf')
                        
                        if mcs_feasible or uav_feasible:
                            if uav_cost <= mcs_cost:
                                solution.add_uav_route(new_uav)
                            else:
                                solution.add_mcs_route(new_mcs)
                            inserted = True
                    else:
                        nearest_centroid = self.problem._find_nearest_centroid(node)
                        new_mcs = Route(vehicle_type='mcs', vehicle_id=len(solution.mcs_routes))
                        new_mcs.start_node = nearest_centroid
                        new_mcs.add_node(node)
                        if self.problem.evaluate_route(new_mcs):
                            solution.add_mcs_route(new_mcs)
                            inserted = True
                            
                    if not inserted:
                        still_unassigned.append(node)
                
                pool.clear()
                break
            
            # 執行插入
            if best_insert_type == 'mcs':
                route = solution.mcs_routes[best_insert_route_idx]
                route.insert_node(best_insert_pos, best_node)
                if not self.problem.evaluate_route(route):
                    route.remove_node(best_insert_pos)
                    self.problem.evaluate_route(route)
                    still_unassigned.append(best_node)
            elif best_insert_type == 'uav':
                route = solution.uav_routes[best_insert_route_idx]
                route.insert_node(best_insert_pos, best_node)
                if not self.problem.evaluate_route(route):
                    route.remove_node(best_insert_pos)
                    self.problem.evaluate_route(route)
                    still_unassigned.append(best_node)
            
            pool.pop(best_node_idx)
        
        solution.unassigned_nodes = still_unassigned