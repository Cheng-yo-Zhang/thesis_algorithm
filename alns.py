"""
ALNS (Adaptive Large Neighborhood Search) — 含車隊平衡機制
===========================================================
Destroy-Repair 迭代改善啟發式，搭配 Simulated Annealing 接受機制
與 adaptive operator weight 自適應選擇。

Destroy Operators:
  1. Random Removal
  2. Worst Removal (cost contribution)
  3. Shaw Removal (relatedness)
  4. Overload Removal (fleet balancing)

Repair Operators:
  1. Greedy Insertion (load-aware)
  2. Regret-2 Insertion (load-aware)
"""

import math
import random
from typing import List, Tuple, Optional, Dict

from config import Config
from models import Node, Route, Solution
from problem import ChargingSchedulingProblem


class ALNSSolver:
    """Adaptive Large Neighborhood Search with fleet balancing."""

    def __init__(self, problem: ChargingSchedulingProblem, cfg: Config):
        self.problem = problem
        self.cfg = cfg

    # ================================================================
    #  主迴圈
    # ================================================================
    def solve(self, initial_solution: Solution,
              dispatch_window_end: float = None) -> Solution:
        cfg = self.cfg
        current = initial_solution.copy()
        best = current.copy()
        T = cfg.SA_INITIAL_TEMP

        total_customers = (len(current.unassigned_nodes)
                           + sum(len(r.nodes) for r in current.get_all_routes()))
        if total_customers == 0:
            return best

        current.calculate_total_cost(total_customers)
        best.calculate_total_cost(total_customers)
        initial_unassigned = len(initial_solution.unassigned_nodes)

        # Operators
        destroy_ops = [
            self._random_removal,
            self._worst_removal,
            self._shaw_removal,
            self._overload_removal,
        ]
        repair_ops = [
            self._greedy_repair,
            self._regret2_repair,
        ]
        n_d, n_r = len(destroy_ops), len(repair_ops)

        d_weights = [1.0] * n_d
        r_weights = [1.0] * n_r
        d_scores = [0.0] * n_d
        r_scores = [0.0] * n_r
        d_counts = [0] * n_d
        r_counts = [0] * n_r

        for i in range(cfg.ALNS_MAX_ITERATIONS):
            candidate = current.copy()

            # 動態移除數量
            removable_count = sum(len(r.nodes) for r in candidate.get_all_routes())
            if removable_count == 0:
                T *= cfg.SA_COOLING_RATE
                continue
            q_max = min(cfg.ALNS_REMOVAL_MAX, removable_count)
            q_min = min(cfg.ALNS_REMOVAL_MIN, q_max)
            q = random.randint(q_min, q_max)

            # 選 operator (roulette wheel)
            d_idx = self._select_operator(d_weights)
            r_idx = self._select_operator(r_weights)

            removed = destroy_ops[d_idx](candidate, q)
            if not removed:
                T *= cfg.SA_COOLING_RATE
                continue

            # 將 unassigned nodes 也納入 repair pool，讓 ALNS 有機會插入
            prev_unassigned = list(candidate.unassigned_nodes)
            candidate.unassigned_nodes = []
            repair_pool = removed + prev_unassigned
            repair_ops[r_idx](candidate, repair_pool, dispatch_window_end)
            candidate.calculate_total_cost(total_customers)

            # Coverage protection
            coverage_loss = len(candidate.unassigned_nodes) - initial_unassigned
            if coverage_loss > cfg.ALNS_MAX_COVERAGE_LOSS:
                T *= cfg.SA_COOLING_RATE
                d_counts[d_idx] += 1
                r_counts[r_idx] += 1
                continue

            # SA acceptance
            delta = candidate.total_cost - current.total_cost
            accepted = False
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                accepted = True
                score = 0.0
                if candidate.total_cost < best.total_cost:
                    best = candidate.copy()
                    score = cfg.ALNS_SIGMA1
                elif delta < 0:
                    score = cfg.ALNS_SIGMA2
                else:
                    score = cfg.ALNS_SIGMA3

                current = candidate
                d_scores[d_idx] += score
                r_scores[r_idx] += score

            d_counts[d_idx] += 1
            r_counts[r_idx] += 1

            # Segment 結束 → 更新權重
            if (i + 1) % cfg.ALNS_SEGMENT_SIZE == 0:
                self._update_weights(d_weights, d_scores, d_counts,
                                     cfg.ALNS_REACTION_FACTOR)
                self._update_weights(r_weights, r_scores, r_counts,
                                     cfg.ALNS_REACTION_FACTOR)
                d_scores = [0.0] * n_d
                r_scores = [0.0] * n_r
                d_counts = [0] * n_d
                r_counts = [0] * n_r

            T *= cfg.SA_COOLING_RATE

        return best

    # ================================================================
    #  Operator Selection & Weight Update
    # ================================================================
    def _select_operator(self, weights: List[float]) -> int:
        total = sum(weights)
        r = random.random() * total
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return i
        return len(weights) - 1

    def _update_weights(self, weights: List[float], scores: List[float],
                        counts: List[int], rho: float) -> None:
        for i in range(len(weights)):
            if counts[i] > 0:
                weights[i] = weights[i] * (1 - rho) + rho * (scores[i] / counts[i])
            weights[i] = max(weights[i], 0.1)

    # ================================================================
    #  Helper: 取得可移除節點
    # ================================================================
    def _get_removable_nodes(self, solution: Solution
                             ) -> List[Tuple[Node, str, int, int]]:
        """回傳 [(node, route_type, route_idx, node_pos), ...]"""
        result = []
        for r_idx, r in enumerate(solution.mcs_routes):
            for n_pos, node in enumerate(r.nodes):
                result.append((node, 'mcs', r_idx, n_pos))
        for r_idx, r in enumerate(solution.uav_routes):
            for n_pos, node in enumerate(r.nodes):
                result.append((node, 'uav', r_idx, n_pos))
        return result

    def _execute_removal(self, solution: Solution,
                         selected: List[Tuple[Node, str, int, int]]
                         ) -> List[Node]:
        """從 solution 中移除 selected nodes，回傳被移除的 node list。"""
        by_route: Dict[Tuple[str, int], List[Tuple[int, Node]]] = {}
        for (node, rtype, r_idx, n_pos) in selected:
            key = (rtype, r_idx)
            by_route.setdefault(key, []).append((n_pos, node))

        removed_nodes = []
        for (rtype, r_idx), items in by_route.items():
            route = (solution.mcs_routes[r_idx] if rtype == 'mcs'
                     else solution.uav_routes[r_idx])
            for n_pos, node in sorted(items, key=lambda x: x[0], reverse=True):
                route.remove_node(n_pos)
                removed_nodes.append(node)
            self.problem.evaluate_route(route)

        return removed_nodes

    def _get_route(self, solution: Solution, rtype: str, r_idx: int) -> Route:
        return (solution.mcs_routes[r_idx] if rtype == 'mcs'
                else solution.uav_routes[r_idx])

    # ================================================================
    #  Destroy Operators
    # ================================================================
    def _random_removal(self, solution: Solution, q: int) -> List[Node]:
        removable = self._get_removable_nodes(solution)
        if not removable:
            return []
        q = min(q, len(removable))
        selected = random.sample(removable, q)
        return self._execute_removal(solution, selected)

    def _worst_removal(self, solution: Solution, q: int) -> List[Node]:
        removable = self._get_removable_nodes(solution)
        if not removable:
            return []

        costs = []
        for (node, rtype, r_idx, n_pos) in removable:
            route = self._get_route(solution, rtype, r_idx)
            candidate = route.copy()
            candidate.remove_node(n_pos)
            self.problem.evaluate_route(candidate)
            saving = route.total_time - candidate.total_time
            costs.append((saving, node, rtype, r_idx, n_pos))

        costs.sort(reverse=True)

        selected = []
        remaining = list(costs)
        for _ in range(min(q, len(remaining))):
            idx = int(random.random() ** 3 * len(remaining))
            item = remaining.pop(idx)
            selected.append((item[1], item[2], item[3], item[4]))

        return self._execute_removal(solution, selected)

    def _shaw_removal(self, solution: Solution, q: int) -> List[Node]:
        removable = self._get_removable_nodes(solution)
        if not removable:
            return []

        seed = random.choice(removable)
        seed_node = seed[0]

        relatedness = []
        for item in removable:
            n = item[0]
            dist = abs(n.x - seed_node.x) + abs(n.y - seed_node.y)
            time_diff = abs(n.due_date - seed_node.due_date) / 60.0
            score = dist + 5.0 * time_diff
            relatedness.append((score, item))

        relatedness.sort(key=lambda x: x[0])

        selected = []
        for _ in range(min(q, len(relatedness))):
            idx = int(random.random() ** 3 * len(relatedness))
            _, item = relatedness.pop(idx)
            selected.append(item)

        return self._execute_removal(solution, selected)

    def _overload_removal(self, solution: Solution, q: int) -> List[Node]:
        """從最繁忙的路線中移除節點 — 專為平衡設計。"""
        route_loads = []
        for r_idx, r in enumerate(solution.mcs_routes):
            if len(r.nodes) > 0:
                route_loads.append(('mcs', r_idx, len(r.nodes)))
        for r_idx, r in enumerate(solution.uav_routes):
            if len(r.nodes) > 0:
                route_loads.append(('uav', r_idx, len(r.nodes)))

        if not route_loads:
            return []

        route_loads.sort(key=lambda x: x[2], reverse=True)

        removed = []
        for rtype, r_idx, n_nodes in route_loads:
            if len(removed) >= q:
                break
            route = self._get_route(solution, rtype, r_idx)
            if len(route.nodes) == 0:
                continue
            n_to_remove = min(q - len(removed), max(1, len(route.nodes) // 3))
            indices = random.sample(range(len(route.nodes)),
                                    min(n_to_remove, len(route.nodes)))
            for idx in sorted(indices, reverse=True):
                node = route.remove_node(idx)
                removed.append(node)
            self.problem.evaluate_route(route)

        return removed

    # ================================================================
    #  Repair Operators
    # ================================================================
    def _greedy_repair(self, solution: Solution, removed: List[Node],
                       dispatch_window_end: float = None) -> None:
        uninserted = list(removed)
        random.shuffle(uninserted)

        still_unassigned = []
        for node in uninserted:
            best = None  # (adjusted_cost, route_type, route_idx, pos)

            for r_idx, route in enumerate(solution.mcs_routes):
                for pos in range(len(route.nodes) + 1):
                    feasible, delta = self.problem.incremental_insertion_check(
                        route, pos, node)
                    if feasible:
                        bonus = self.problem._get_type_preference_bonus(
                            route.vehicle_type, node, dispatch_window_end)
                        load_pen = self._load_penalty(route, solution)
                        adj = delta - bonus + load_pen
                        if best is None or adj < best[0]:
                            best = (adj, 'mcs', r_idx, pos)

            if (best is None
                    and node.node_type == 'urgent'
                    and self.problem.compute_uav_delivery(node) > 0):
                for r_idx, route in enumerate(solution.uav_routes):
                    for pos in range(len(route.nodes) + 1):
                        feasible, delta = self.problem.incremental_insertion_check(
                            route, pos, node)
                        if feasible:
                            bonus = self.problem._get_type_preference_bonus(
                                'uav', node, dispatch_window_end)
                            load_pen = self._load_penalty(route, solution)
                            adj = delta - bonus + load_pen
                            if best is None or adj < best[0]:
                                best = (adj, 'uav', r_idx, pos)

            if best is not None:
                _, rtype, r_idx, pos = best
                route = self._get_route(solution, rtype, r_idx)
                route.insert_node(pos, node)
                self.problem.evaluate_route(route)
                node.status = 'assigned'
            else:
                still_unassigned.append(node)

        solution.unassigned_nodes.extend(still_unassigned)

    def _regret2_repair(self, solution: Solution, removed: List[Node],
                        dispatch_window_end: float = None) -> None:
        pool = list(removed)
        still_unassigned = []

        while pool:
            best_priority = None
            best_node = None
            best_node_idx = -1
            best_insertion = None  # (adjusted_cost, route_type, route_idx, pos)

            for n_idx, node in enumerate(pool):
                options = []

                for r_idx, route in enumerate(solution.mcs_routes):
                    for pos in range(len(route.nodes) + 1):
                        feasible, delta = self.problem.incremental_insertion_check(
                            route, pos, node)
                        if feasible:
                            bonus = self.problem._get_type_preference_bonus(
                                route.vehicle_type, node, dispatch_window_end)
                            load_pen = self._load_penalty(route, solution)
                            adj = delta - bonus + load_pen
                            options.append((adj, 'mcs', r_idx, pos))

                if (not options
                        and node.node_type == 'urgent'
                        and self.problem.compute_uav_delivery(node) > 0):
                    for r_idx, route in enumerate(solution.uav_routes):
                        for pos in range(len(route.nodes) + 1):
                            feasible, delta = self.problem.incremental_insertion_check(
                                route, pos, node)
                            if feasible:
                                bonus = self.problem._get_type_preference_bonus(
                                    'uav', node, dispatch_window_end)
                                load_pen = self._load_penalty(route, solution)
                                adj = delta - bonus + load_pen
                                options.append((adj, 'uav', r_idx, pos))

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
                    best_insertion = options[0]

            if best_node is None:
                still_unassigned.extend(pool)
                break

            _, rtype, r_idx, pos = best_insertion
            route = self._get_route(solution, rtype, r_idx)
            route.insert_node(pos, best_node)
            if self.problem.evaluate_route(route):
                best_node.status = 'assigned'
            else:
                route.remove_node(pos)
                self.problem.evaluate_route(route)
                still_unassigned.append(best_node)

            pool.pop(best_node_idx)

        solution.unassigned_nodes.extend(still_unassigned)

    # ================================================================
    #  Load-Aware Penalty（平衡輔助）
    # ================================================================
    def _load_penalty(self, route: Route, solution: Solution) -> float:
        all_routes = solution.get_all_routes()
        total_nodes = sum(len(r.nodes) for r in all_routes)
        total_routes = len(all_routes)
        if total_routes == 0:
            return 0.0
        avg_load = total_nodes / total_routes
        if avg_load == 0:
            return 0.0
        ratio = len(route.nodes) / avg_load
        if ratio <= 1.0:
            return 0.0
        return self.cfg.LAMBDA_BALANCE * 0.1 * (ratio - 1.0)

    # ================================================================
    #  Flexible Greedy Insertion (for multi-slot aggregation)
    # ================================================================
    def _flexible_greedy_insertion(self, solution: Solution,
                                   extra_nodes: List[Node]) -> None:
        """將 extra_nodes 用 greedy 方式插入現有 solution。
        供 compare_gi_alns.py 的 multi-slot aggregation 使用。
        """
        self._greedy_repair(solution, extra_nodes, dispatch_window_end=None)

    # ================================================================
    #  Cross-Fleet Local Search（可選）
    # ================================================================
    def cross_fleet_local_search(self, solution: Solution) -> Solution:
        """嘗試在不同車型的 MCS 路線之間交換節點以改善 cost。"""
        improved = True
        while improved:
            improved = False
            for i, r1 in enumerate(solution.mcs_routes):
                if len(r1.nodes) == 0:
                    continue
                for j, r2 in enumerate(solution.mcs_routes):
                    if i >= j or r1.vehicle_type == r2.vehicle_type:
                        continue
                    if len(r2.nodes) == 0:
                        continue
                    for p1 in range(len(r1.nodes)):
                        for p2 in range(len(r2.nodes)):
                            old_cost = r1.total_time + r2.total_time
                            n1, n2 = r1.nodes[p1], r2.nodes[p2]
                            # swap
                            r1.nodes[p1], r2.nodes[p2] = n2, n1
                            r1.total_demand += (n2.demand - n1.demand)
                            r2.total_demand += (n1.demand - n2.demand)
                            f1 = self.problem.evaluate_route(r1)
                            f2 = self.problem.evaluate_route(r2)
                            if (f1 and f2
                                    and (r1.total_time + r2.total_time) < old_cost):
                                improved = True
                            else:
                                r1.nodes[p1], r2.nodes[p2] = n1, n2
                                r1.total_demand += (n1.demand - n2.demand)
                                r2.total_demand += (n2.demand - n1.demand)
                                self.problem.evaluate_route(r1)
                                self.problem.evaluate_route(r2)

        solution.calculate_total_cost()
        return solution
