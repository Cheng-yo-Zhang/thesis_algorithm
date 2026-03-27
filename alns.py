import random
from typing import List, Dict, Callable

import numpy as np

from config import Config
from models import Node, Route, Solution
from problem import ChargingSchedulingProblem


class ALNSSolver:
    """ALNS 求解器 (保留, 可用於 slot 內改善)"""

    def __init__(self, problem: ChargingSchedulingProblem, cfg: Config = None):
        self.problem = problem
        self.cfg = cfg or problem.cfg

        self.destroy_ops: Dict[str, Callable] = {
            'random_removal':  self._random_removal,
            'worst_removal':   self._worst_removal,
            'shaw_removal':    self._shaw_removal,
            'urgency_removal': self._urgency_removal,
            'route_removal':   self._route_removal,
        }
        self.repair_ops: Dict[str, Callable] = {
            'flexible_greedy':  self._flexible_greedy_insertion,
            'flexible_regret2': self._flexible_regret2_insertion,
            'regret3':          self._regret3_insertion,
        }
        self.destroy_weights: Dict[str, float] = {k: 1.0 for k in self.destroy_ops}
        self.repair_weights:  Dict[str, float] = {k: 1.0 for k in self.repair_ops}
        self._reset_segment_stats()

    def _count_completable(self, solution: Solution, window_end: float) -> int:
        """Count how many assigned customers can finish service within window_end."""
        count = 0
        for route in solution.get_all_routes():
            for i, node in enumerate(route.nodes):
                if i < len(route.departure_times) and route.departure_times[i] <= window_end:
                    count += 1
        return count

    def solve(self, initial_solution: Solution, dispatch_window_end: float = None) -> Solution:
        c = self.cfg
        current = initial_solution.copy()
        best = initial_solution.copy()
        temperature = c.SA_INITIAL_TEMP
        total_customers = current.total_customers

        served_count = sum(len(r.nodes) for r in current.get_all_routes())
        min_remove = max(1, int(served_count * c.DESTROY_RATIO_MIN))
        max_remove = max(min_remove + 1, int(served_count * c.DESTROY_RATIO_MAX))
        # Allow meaningful restructuring (10-40% of assigned nodes)
        max_remove = max(min_remove + 1, int(served_count * c.DESTROY_RATIO_MAX))

        current_unassigned = len(current.unassigned_nodes)
        if dispatch_window_end:
            current_completable = self._count_completable(current, dispatch_window_end)

        for iteration in range(1, c.ALNS_MAX_ITERATIONS + 1):
            d_name = self._roulette_select(self.destroy_weights)
            r_name = self._roulette_select(self.repair_weights)

            candidate = current.copy()
            num_remove = random.randint(min_remove, max_remove)
            removed = self.destroy_ops[d_name](candidate, num_remove)
            self.repair_ops[r_name](candidate, removed)
            # Re-evaluate all routes to get updated departure_times
            for route in candidate.get_all_routes():
                self.problem.evaluate_route(route)
            candidate.calculate_total_cost(total_customers)

            # Reject solutions that lose customers (more unassigned)
            cand_unassigned = len(candidate.unassigned_nodes)
            if cand_unassigned > current_unassigned:
                continue

            # Reject solutions that reduce completable customers within window
            if dispatch_window_end:
                cand_completable = self._count_completable(candidate, dispatch_window_end)
                if cand_completable < current_completable:
                    continue

            delta = candidate.total_cost - current.total_cost
            if delta < -1e-9:
                current = candidate
                current_unassigned = cand_unassigned
                if dispatch_window_end:
                    current_completable = cand_completable
                if current.total_cost < best.total_cost - 1e-9:
                    best = current.copy()
                    self._record_score(d_name, r_name, c.SIGMA_1)
                else:
                    self._record_score(d_name, r_name, c.SIGMA_2)
            elif self._sa_accept(delta, temperature):
                current = candidate
                current_unassigned = cand_unassigned
                if dispatch_window_end:
                    current_completable = cand_completable
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

    # --- Helper: collect all assigned node candidates ---
    def _collect_candidates(self, solution: Solution) -> List[tuple]:
        """Collect (route_type, route_idx, node_pos) for all assigned nodes across MCS + UAV."""
        candidates = []
        for r_idx, r in enumerate(solution.mcs_routes):
            for n_pos in range(len(r.nodes)):
                candidates.append(('mcs', r_idx, n_pos))
        for r_idx, r in enumerate(solution.uav_routes):
            for n_pos in range(len(r.nodes)):
                candidates.append(('uav', r_idx, n_pos))
        return candidates

    def _execute_removals(self, solution: Solution, to_remove: List[tuple]) -> List[Node]:
        """Execute node removals sorted by (route_type, route_idx, node_pos DESC) to avoid index shift."""
        to_remove.sort(key=lambda x: (x[0], x[1], -x[2] if len(x) > 2 else 0), reverse=False)
        # Group by (route_type, route_idx) and sort positions descending within each group
        from collections import defaultdict
        groups = defaultdict(list)
        for item in to_remove:
            v_type, r_idx = item[0], item[1]
            n_pos = item[2] if len(item) > 2 else item[-1]
            groups[(v_type, r_idx)].append(n_pos)
        removed = []
        for (v_type, r_idx), positions in groups.items():
            routes = solution.mcs_routes if v_type == 'mcs' else solution.uav_routes
            for n_pos in sorted(positions, reverse=True):
                removed.append(routes[r_idx].remove_node(n_pos))
        for r in solution.get_all_routes():
            self.problem.evaluate_route(r)
        return removed

    # --- Destroy ---
    def _random_removal(self, solution: Solution, num_remove: int) -> List[Node]:
        candidates = self._collect_candidates(solution)
        if not candidates:
            return []
        num_remove = min(num_remove, len(candidates))
        selected = random.sample(candidates, num_remove)
        return self._execute_removals(solution, selected)

    def _worst_removal(self, solution: Solution, num_remove: int) -> List[Node]:
        node_costs = []
        for r_idx, r in enumerate(solution.mcs_routes):
            for n_pos, node in enumerate(r.nodes):
                cost = r.user_waiting_times[n_pos] if n_pos < len(r.user_waiting_times) else 0.0
                node_costs.append((cost, 'mcs', r_idx, n_pos))
        for r_idx, r in enumerate(solution.uav_routes):
            for n_pos, node in enumerate(r.nodes):
                cost = r.user_waiting_times[n_pos] if n_pos < len(r.user_waiting_times) else 0.0
                node_costs.append((cost, 'uav', r_idx, n_pos))
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
        return self._execute_removals(solution, [(v, r, p) for (_, v, r, p) in to_remove])

    def _shaw_removal(self, solution: Solution, num_remove: int) -> List[Node]:
        """Shaw Relatedness Removal — 移除空間/時間/需求相似的節點群。"""
        c = self.cfg
        all_assigned = solution.get_all_assigned_nodes_with_positions()
        if not all_assigned:
            return []

        # Select random seed
        seed_idx = random.randint(0, len(all_assigned) - 1)
        seed_node, seed_vtype, seed_ridx, seed_npos = all_assigned[seed_idx]

        # Compute max values for normalization
        max_dist = c.AREA_SIZE * 2.0  # max Manhattan distance in grid
        max_demand = 20.0  # upper bound of demand

        # Compute relatedness for all other nodes
        relatedness = []
        for node, v_type, r_idx, n_pos in all_assigned:
            if node.id == seed_node.id:
                continue
            dist = abs(seed_node.x - node.x) + abs(seed_node.y - node.y)
            r = (c.SHAW_W_DIST * dist / max_dist
                 + c.SHAW_W_TIME * abs(seed_node.ready_time - node.ready_time) / c.T_TOTAL
                 + c.SHAW_W_DEMAND * abs(seed_node.demand - node.demand) / max_demand
                 + c.SHAW_W_TYPE * (0.0 if v_type == seed_vtype else 1.0))
            relatedness.append((r, v_type, r_idx, n_pos))

        relatedness.sort(key=lambda x: x[0])  # most related (smallest) first
        to_remove = [(seed_vtype, seed_ridx, seed_npos)]
        num_remove = min(num_remove, len(relatedness) + 1)

        for _ in range(num_remove - 1):
            if not relatedness:
                break
            idx = int(random.random() ** c.SHAW_REMOVAL_P * len(relatedness))
            idx = min(idx, len(relatedness) - 1)
            _, v_type, r_idx, n_pos = relatedness.pop(idx)
            to_remove.append((v_type, r_idx, n_pos))

        return self._execute_removals(solution, to_remove)

    def _urgency_removal(self, solution: Solution, num_remove: int) -> List[Node]:
        """Urgency Removal — 移除 slack 最緊的節點 (due_date - departure_time)，重新分配給更快車輛。"""
        node_slacks = []
        for r_idx, r in enumerate(solution.mcs_routes):
            for n_pos, node in enumerate(r.nodes):
                dep = r.departure_times[n_pos] if n_pos < len(r.departure_times) else node.ready_time
                slack = node.due_date - dep
                node_slacks.append((slack, 'mcs', r_idx, n_pos))
        for r_idx, r in enumerate(solution.uav_routes):
            for n_pos, node in enumerate(r.nodes):
                dep = r.departure_times[n_pos] if n_pos < len(r.departure_times) else node.ready_time
                slack = node.due_date - dep
                node_slacks.append((slack, 'uav', r_idx, n_pos))

        if not node_slacks:
            return []
        node_slacks.sort(key=lambda x: x[0])  # tightest slack first
        to_remove = []
        num_remove = min(num_remove, len(node_slacks))
        for _ in range(num_remove):
            if not node_slacks:
                break
            idx = int(random.random() ** self.cfg.WORST_REMOVAL_P * len(node_slacks))
            idx = min(idx, len(node_slacks) - 1)
            to_remove.append(node_slacks.pop(idx))
        return self._execute_removals(solution, [(v, r, p) for (_, v, r, p) in to_remove])

    def _route_removal(self, solution: Solution, num_remove: int) -> List[Node]:
        """Route Removal — 移除短路徑 (1-2 nodes) 的所有節點，釋放車輛資源。"""
        short_routes = []
        for r_idx, r in enumerate(solution.mcs_routes):
            if 0 < len(r.nodes) <= 2:
                short_routes.append(('mcs', r_idx, len(r.nodes)))
        for r_idx, r in enumerate(solution.uav_routes):
            if 0 < len(r.nodes) <= 2:
                short_routes.append(('uav', r_idx, len(r.nodes)))

        if not short_routes:
            # Fallback to random removal
            return self._random_removal(solution, num_remove)

        # Sort by route length (shortest first)
        short_routes.sort(key=lambda x: x[2])
        to_remove = []
        for v_type, r_idx, n_count in short_routes:
            if len(to_remove) >= num_remove:
                break
            routes = solution.mcs_routes if v_type == 'mcs' else solution.uav_routes
            for n_pos in range(n_count):
                to_remove.append((v_type, r_idx, n_pos))

        return self._execute_removals(solution, to_remove[:num_remove])

    # --- Repair ---
    def _find_best_insertion_all_routes(self, solution: Solution, node: Node):
        """跨車隊搜尋最佳插入位置 — 同時搜尋 MCS + UAV 路徑。
        Returns: (route_collection, route_idx, position, cost) or None
        """
        best_collection, best_idx, best_pos, min_cost = None, -1, -1, float('inf')

        # Search all MCS routes
        for r_idx, r in enumerate(solution.mcs_routes):
            for pos in range(len(r.nodes) + 1):
                ok, dc = self.problem.incremental_insertion_check(r, pos, node)
                if ok and dc < min_cost:
                    min_cost, best_collection, best_idx, best_pos = dc, 'mcs', r_idx, pos

        # Search all UAV routes (UAV 僅服務 urgent)
        if node.node_type == 'urgent' and not node.uav_served and self.problem.compute_uav_delivery(node) > 0:
            for r_idx, r in enumerate(solution.uav_routes):
                for pos in range(len(r.nodes) + 1):
                    ok, dc = self.problem.incremental_insertion_check(r, pos, node)
                    if ok and dc < min_cost:
                        min_cost, best_collection, best_idx, best_pos = dc, 'uav', r_idx, pos

        if best_collection is not None:
            return best_collection, best_idx, best_pos, min_cost
        return None

    def _insert_into_solution(self, solution: Solution, node: Node,
                               collection: str, r_idx: int, pos: int) -> bool:
        """將節點插入指定路徑，驗證可行性。失敗則回滾。"""
        routes = solution.mcs_routes if collection == 'mcs' else solution.uav_routes
        routes[r_idx].insert_node(pos, node)
        if self.problem.evaluate_route(routes[r_idx]):
            return True
        routes[r_idx].remove_node(pos)
        self.problem.evaluate_route(routes[r_idx])
        return False

    def _flexible_greedy_insertion(self, solution: Solution, removed_nodes: List[Node]):
        """Cross-Fleet Greedy Insertion — 搜尋所有車隊 (MCS + UAV) 的最佳插入位置。"""
        pool = list(removed_nodes)
        pool.sort(key=lambda n: (0 if n.node_type == 'urgent' else 1, n.due_date))
        still_unassigned = []
        for node in pool:
            result = self._find_best_insertion_all_routes(solution, node)
            if result is not None:
                collection, r_idx, pos, cost = result
                if not self._insert_into_solution(solution, node, collection, r_idx, pos):
                    still_unassigned.append(node)
            else:
                still_unassigned.append(node)
        solution.unassigned_nodes = still_unassigned

    def _collect_all_options(self, solution: Solution, node: Node) -> List[tuple]:
        """收集節點在所有路徑 (MCS + UAV) 的可行插入選項 [(cost, collection, r_idx, pos), ...]。"""
        options = []
        for r_idx, r in enumerate(solution.mcs_routes):
            for pos in range(len(r.nodes) + 1):
                ok, dc = self.problem.incremental_insertion_check(r, pos, node)
                if ok:
                    options.append((dc, 'mcs', r_idx, pos))
        if node.node_type == 'urgent' and not node.uav_served and self.problem.compute_uav_delivery(node) > 0:
            for r_idx, r in enumerate(solution.uav_routes):
                for pos in range(len(r.nodes) + 1):
                    ok, dc = self.problem.incremental_insertion_check(r, pos, node)
                    if ok:
                        options.append((dc, 'uav', r_idx, pos))
        options.sort(key=lambda x: x[0])
        return options

    def _flexible_regret2_insertion(self, solution: Solution, removed_nodes: List[Node]):
        """Cross-Fleet Regret-2 Insertion — 跨車隊 regret 值計算。"""
        self._regret_k_insertion(solution, removed_nodes, k=2)

    def _regret3_insertion(self, solution: Solution, removed_nodes: List[Node]):
        """Cross-Fleet Regret-3 Insertion — 更深層的前瞻式插入。"""
        self._regret_k_insertion(solution, removed_nodes, k=3)

    def _regret_k_insertion(self, solution: Solution, removed_nodes: List[Node], k: int = 2):
        """Generalized Regret-k Insertion across all vehicle types."""
        pool = list(removed_nodes)
        still_unassigned = []
        while pool:
            best_node, best_n_idx = None, -1
            best_regret = -float('inf')
            best_collection, best_idx, best_pos = None, -1, -1
            best_cost = float('inf')

            for n_idx, node in enumerate(pool):
                options = self._collect_all_options(solution, node)
                if not options:
                    continue

                c1, col1, r1, p1 = options[0]
                # Regret = sum of (c_i - c_1) for i=2..k
                regret = 0.0
                for i in range(1, min(k, len(options))):
                    regret += options[i][0] - c1
                if len(options) < k:
                    regret += (k - len(options)) * 1e6  # high regret for scarce options

                if (regret > best_regret + 1e-9 or
                        (abs(regret - best_regret) < 1e-9 and c1 < best_cost - 1e-9)):
                    best_regret = regret
                    best_node, best_n_idx = node, n_idx
                    best_collection, best_idx, best_pos = col1, r1, p1
                    best_cost = c1

            if best_node is None:
                still_unassigned.extend(pool)
                break

            if not self._insert_into_solution(solution, best_node, best_collection, best_idx, best_pos):
                still_unassigned.append(best_node)
            pool.pop(best_n_idx)

        solution.unassigned_nodes = still_unassigned

    # ==================== Cross-Fleet Local Search ====================
    def cross_fleet_local_search(self, solution: Solution) -> Solution:
        """Phase 3: 跨車隊局部搜尋 — transfer + swap + unassigned rescue。"""
        c = self.cfg
        total_customers = solution.total_customers

        solution = self._cross_fleet_transfer(solution, total_customers)
        solution = self._cross_fleet_swap(solution, total_customers)
        solution = self._unassigned_rescue(solution, total_customers)

        solution.calculate_total_cost(total_customers)
        return solution

    def _cross_fleet_transfer(self, solution: Solution, total_customers: int) -> Solution:
        """Inter-Type Transfer — 嘗試將節點從一種車隊轉移到另一種車隊。"""
        c = self.cfg
        improved = True
        iterations = 0
        while improved and iterations < c.CROSS_FLEET_MAX_ITER:
            improved = False
            iterations += 1
            solution.calculate_total_cost(total_customers)
            old_cost = solution.total_cost

            all_assigned = solution.get_all_assigned_nodes_with_positions()
            random.shuffle(all_assigned)

            for node, src_type, src_ridx, src_npos in all_assigned:
                # Determine destination route collection
                if src_type == 'mcs':
                    # Try MCS→UAV transfer
                    if node.node_type != 'urgent' or node.uav_served or self.problem.compute_uav_delivery(node) <= 0:
                        # Also try MCS-SLOW↔MCS-FAST transfer
                        src_route = solution.mcs_routes[src_ridx]
                        other_mcs_type = 'mcs_fast' if src_route.vehicle_type == 'mcs_slow' else 'mcs_slow'
                        dst_routes_list = []
                        for r_idx, r in enumerate(solution.mcs_routes):
                            if r.vehicle_type == other_mcs_type:
                                dst_routes_list.append(('mcs', r_idx, r))
                    else:
                        dst_routes_list = [('uav', r_idx, r) for r_idx, r in enumerate(solution.uav_routes)]
                        src_route = solution.mcs_routes[src_ridx]
                        other_mcs_type = 'mcs_fast' if src_route.vehicle_type == 'mcs_slow' else 'mcs_slow'
                        for r_idx, r in enumerate(solution.mcs_routes):
                            if r.vehicle_type == other_mcs_type:
                                dst_routes_list.append(('mcs', r_idx, r))
                else:
                    # UAV→MCS transfer
                    dst_routes_list = [('mcs', r_idx, r) for r_idx, r in enumerate(solution.mcs_routes)]

                for dst_collection, dst_ridx, dst_route in dst_routes_list:
                    # Skip same route
                    if dst_collection == src_type and dst_ridx == src_ridx:
                        continue

                    # Find best insertion position in dst
                    best_pos, best_dc = -1, float('inf')
                    for pos in range(len(dst_route.nodes) + 1):
                        ok, dc = self.problem.incremental_insertion_check(dst_route, pos, node)
                        if ok and dc < best_dc:
                            best_dc, best_pos = dc, pos

                    if best_pos < 0:
                        continue

                    # Tentative transfer: remove from src, insert into dst
                    src_routes = solution.mcs_routes if src_type == 'mcs' else solution.uav_routes
                    if src_npos >= len(src_routes[src_ridx].nodes):
                        break
                    if src_routes[src_ridx].nodes[src_npos].id != node.id:
                        break  # Index shifted, skip

                    src_routes[src_ridx].remove_node(src_npos)
                    self.problem.evaluate_route(src_routes[src_ridx])

                    dst_routes = solution.mcs_routes if dst_collection == 'mcs' else solution.uav_routes
                    dst_routes[dst_ridx].insert_node(best_pos, node)
                    if self.problem.evaluate_route(dst_routes[dst_ridx]):
                        solution.calculate_total_cost(total_customers)
                        if solution.total_cost < old_cost - 1e-9:
                            improved = True
                            break  # Restart scan with updated positions
                        else:
                            # No improvement — rollback
                            dst_routes[dst_ridx].remove_node(best_pos)
                            self.problem.evaluate_route(dst_routes[dst_ridx])
                            src_routes[src_ridx].insert_node(src_npos, node)
                            self.problem.evaluate_route(src_routes[src_ridx])
                            solution.calculate_total_cost(total_customers)
                    else:
                        # Infeasible — rollback
                        dst_routes[dst_ridx].remove_node(best_pos)
                        self.problem.evaluate_route(dst_routes[dst_ridx])
                        src_routes[src_ridx].insert_node(src_npos, node)
                        self.problem.evaluate_route(src_routes[src_ridx])

                if improved:
                    break  # Restart outer loop

        return solution

    def _cross_fleet_swap(self, solution: Solution, total_customers: int) -> Solution:
        """Inter-Type Swap — 交換不同車隊間的節點對。"""
        c = self.cfg
        improved = True
        iterations = 0
        while improved and iterations < c.CROSS_FLEET_MAX_ITER:
            improved = False
            iterations += 1
            solution.calculate_total_cost(total_customers)
            old_cost = solution.total_cost

            for mcs_ridx, mcs_route in enumerate(solution.mcs_routes):
                if not mcs_route.nodes:
                    continue
                for uav_ridx, uav_route in enumerate(solution.uav_routes):
                    if not uav_route.nodes:
                        continue
                    for mi in range(len(mcs_route.nodes)):
                        mcs_node = mcs_route.nodes[mi]
                        if mcs_node.node_type != 'urgent' or mcs_node.uav_served or self.problem.compute_uav_delivery(mcs_node) <= 0:
                            continue
                        for ui in range(len(uav_route.nodes)):
                            uav_node = uav_route.nodes[ui]

                            # Execute swap
                            mcs_route.remove_node(mi)
                            uav_route.remove_node(ui)
                            mcs_route.insert_node(mi, uav_node)
                            uav_route.insert_node(ui, mcs_node)

                            mcs_ok = self.problem.evaluate_route(mcs_route)
                            uav_ok = self.problem.evaluate_route(uav_route)

                            if mcs_ok and uav_ok:
                                solution.calculate_total_cost(total_customers)
                                if solution.total_cost < old_cost - 1e-9:
                                    improved = True
                                    break
                            # Rollback
                            mcs_route.remove_node(mi)
                            uav_route.remove_node(ui)
                            mcs_route.insert_node(mi, mcs_node)
                            uav_route.insert_node(ui, uav_node)
                            self.problem.evaluate_route(mcs_route)
                            self.problem.evaluate_route(uav_route)

                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return solution

    def _unassigned_rescue(self, solution: Solution, total_customers: int) -> Solution:
        """Unassigned Rescue — 最後嘗試將未分配節點插入任何可行路徑。"""
        if not solution.unassigned_nodes:
            return solution

        remaining = []
        rescue_pool = sorted(solution.unassigned_nodes,
                             key=lambda n: (0 if n.node_type == 'urgent' else 1, n.due_date))
        for node in rescue_pool:
            result = self._find_best_insertion_all_routes(solution, node)
            if result is not None:
                collection, r_idx, pos, cost = result
                if self._insert_into_solution(solution, node, collection, r_idx, pos):
                    continue
            remaining.append(node)

        solution.unassigned_nodes = remaining
        return solution
