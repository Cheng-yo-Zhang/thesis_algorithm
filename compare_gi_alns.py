"""
Greedy Insertion vs GI+ALNS — 覆蓋率比較 + 前 10 slot 路徑圖
=============================================================
輸出到 comparison_gi_alns/ 資料夾
"""

import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import List

from config import Config
from models import Node, VehicleState, Route, Solution
from problem import ChargingSchedulingProblem
from simulation import initialize_fleet, simulate_dispatch_window
from alns import ALNSSolver
from visualization import (
    export_requests_csv, plot_per_slot_mcs_routes, print_terminal_report,
)


def run_with_strategy(strategy: str, cfg: Config, verbose: bool = False):
    """
    用指定策略跑一次完整模擬。
    strategy: "gi_only" | "gi_alns"
    """
    problem = ChargingSchedulingProblem(cfg)
    fleet = initialize_fleet(cfg, problem.depot)

    all_requests: List[Node] = []
    backlog: List[Node] = []

    total_generated = 0
    total_served = 0
    total_missed = 0
    slot_data: List[dict] = []
    num_slots = cfg.T_TOTAL // cfg.DELTA_T

    # Multi-slot ALNS aggregation buffer
    alns_buffer_solutions: List = []
    alns_buffer_window_ends: List[float] = []

    for k in range(1, num_slots + 1):
        if k % 20 == 1 or k == num_slots:
            print(f"    [{strategy}] Slot {k}/{num_slots}...", flush=True)
        slot_start = (k - 1) * cfg.DELTA_T
        slot_end = k * cfg.DELTA_T
        dispatch_time = slot_end
        next_decision_time = slot_end + cfg.DELTA_T

        new_requests = problem.generate_requests(slot_start, slot_number=k)
        total_generated += len(new_requests)

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

        active_pool = new_requests + active_backlog

        # Update committed nodes
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
                vs.available_time = slot_end
        total_served += completed_committed

        if not active_pool:
            slot_data.append({
                "slot": k, "slot_start": slot_start, "slot_end": slot_end,
                "new_requests": [], "active_backlog": [],
                "active_pool": [], "solution": None,
                "n_assigned": 0, "slot_served": completed_committed,
                "slot_missed": slot_missed, "backlog_out": [],
            })
            continue

        all_customer_nodes = [n for n in all_requests if n.status not in ('served', 'missed')]
        all_customer_nodes.extend(new_requests)
        all_requests.extend(new_requests)
        problem.setup_nodes(all_customer_nodes)

        # === 策略分歧點 ===
        # 兩組都先用 Greedy Insertion 建構初始解
        solution = problem.greedy_insertion_construction(
            active_pool, fleet, dispatch_window_end=next_decision_time)

        # GI+ALNS: 在 GI 解的基礎上跑 ALNS 改善
        if strategy == "gi_alns":
            L = getattr(cfg, 'ALNS_LOOKAHEAD_SLOTS', 1)
            assigned_count = sum(len(r.nodes) for r in solution.get_all_routes())

            if L <= 1:
                # Per-slot ALNS
                if assigned_count > 0:
                    solver = ALNSSolver(problem, cfg)
                    solution = solver.solve(solution, dispatch_window_end=next_decision_time)
                    if getattr(cfg, 'ENABLE_CROSS_FLEET_LS', False):
                        solution = solver.cross_fleet_local_search(solution)
            else:
                # Multi-slot aggregated ALNS
                alns_buffer_solutions.append(solution)
                alns_buffer_window_ends.append(next_decision_time)

                if len(alns_buffer_solutions) >= L or k == num_slots:
                    merged = solution.copy()
                    all_assigned_ids = set()
                    for r in merged.get_all_routes():
                        for n in r.nodes:
                            all_assigned_ids.add(n.id)
                    extra_nodes = []
                    for prev_sol in alns_buffer_solutions[:-1]:
                        for r in prev_sol.get_all_routes():
                            for n in r.nodes:
                                if n.id not in all_assigned_ids and n.status == 'assigned':
                                    extra_nodes.append(n)
                                    all_assigned_ids.add(n.id)
                    if extra_nodes:
                        temp_solver = ALNSSolver(problem, cfg)
                        temp_solver._flexible_greedy_insertion(merged, extra_nodes)
                        for r in merged.get_all_routes():
                            problem.evaluate_route(r)

                    farthest_window_end = max(alns_buffer_window_ends)
                    total_assigned = sum(len(r.nodes) for r in merged.get_all_routes())

                    if total_assigned > 0:
                        solver = ALNSSolver(problem, cfg)
                        merged = solver.solve(merged, dispatch_window_end=farthest_window_end)
                        if getattr(cfg, 'ENABLE_CROSS_FLEET_LS', False):
                            merged = solver.cross_fleet_local_search(merged)

                    solution = merged
                    alns_buffer_solutions.clear()
                    alns_buffer_window_ends.clear()

        n_assigned = sum(1 for n in active_pool if n.status == 'assigned')
        snapshot = deepcopy(solution)

        simulate_dispatch_window(solution, fleet, problem, next_decision_time)

        new_backlog: List[Node] = []
        for node in solution.unassigned_nodes:
            if node.due_date > next_decision_time:
                node.status = 'backlog'
                new_backlog.append(node)
            else:
                node.status = 'missed'
                total_missed += 1
                slot_missed += 1

        for route in solution.get_all_routes():
            for node in route.nodes:
                if node.status == 'assigned':
                    node.status = 'backlog'
                    new_backlog.append(node)

        backlog = new_backlog

        # UAV re-queue
        if cfg.UAV_REQUEUE_FOR_MCS:
            for route in solution.uav_routes:
                for node in route.nodes:
                    if node.status == 'served' and node.uav_served and node.residual_demand > 1.0:
                        followup = Node(
                            id=problem._next_node_id,
                            x=node.x, y=node.y,
                            demand=node.residual_demand,
                            ready_time=next_decision_time,
                            due_date=next_decision_time + np.random.uniform(
                                cfg.NORMAL_TW_MIN, cfg.NORMAL_TW_MAX),
                            service_time=0,
                            node_type='normal',
                            status='backlog',
                            ev_current_soc=node.ev_current_soc,
                            original_demand=node.residual_demand,
                            uav_served=True,
                            residual_demand=node.residual_demand,
                            origin_slot=k,
                        )
                        problem._next_node_id += 1
                        backlog.append(followup)
                        total_generated += 1

        direct_served = 0
        for route in solution.get_all_routes():
            for node in route.nodes:
                if node.status == 'served':
                    direct_served += 1
        slot_served = completed_committed + direct_served
        total_served += direct_served

        slot_data.append({
            "slot": k, "slot_start": slot_start, "slot_end": slot_end,
            "new_requests": new_requests, "active_backlog": active_backlog,
            "active_pool": active_pool, "solution": snapshot,
            "n_assigned": n_assigned, "slot_served": slot_served,
            "slot_missed": slot_missed, "backlog_out": new_backlog,
        })

    # Terminal sweep
    terminal_committed = 0
    for vs in fleet:
        for node in vs.committed_nodes:
            node.status = 'served'
            terminal_committed += 1
        vs.committed_nodes = []
        vs.committed_departures = []
    total_served += terminal_committed

    for req in backlog:
        req.status = 'missed'
        total_missed += 1
    backlog_remaining = len(backlog)
    backlog = []

    # Measurement window
    mw_requests = [r for r in all_requests
                   if cfg.MEASUREMENT_START <= r.ready_time < cfg.MEASUREMENT_END]
    mw_total = len(mw_requests)
    mw_served = sum(1 for r in mw_requests if r.status == 'served')
    mw_missed = sum(1 for r in mw_requests if r.status == 'missed')
    mw_service_rate = mw_served / mw_total if mw_total > 0 else 1.0

    mw_urgent = [r for r in mw_requests if r.node_type == 'urgent']
    mw_normal = [r for r in mw_requests if r.node_type == 'normal']
    mw_urgent_served = sum(1 for r in mw_urgent if r.status == 'served')
    mw_normal_served = sum(1 for r in mw_normal if r.status == 'served')
    mw_urgent_rate = mw_urgent_served / len(mw_urgent) if mw_urgent else 1.0
    mw_normal_rate = mw_normal_served / len(mw_normal) if mw_normal else 1.0

    mw_served_requests = [r for r in mw_requests if r.status == 'served']
    mw_vehicles_used = {}
    for r in mw_served_requests:
        vtype = r.served_by_vehicle_type
        vid = r.served_by_vehicle_id
        if vtype and vid >= 0:
            mw_vehicles_used.setdefault(vtype, set()).add(vid)

    mw_n_slow = len(mw_vehicles_used.get('mcs_slow', set()))
    mw_n_fast = len(mw_vehicles_used.get('mcs_fast', set()))
    mw_n_uav = len(mw_vehicles_used.get('uav', set()))

    stats = {
        "total_generated": total_generated,
        "total_served": total_served,
        "total_missed": total_missed,
        "backlog_remaining": backlog_remaining,
        "terminal_committed": terminal_committed,
        "all_requests": all_requests,
        "fleet": fleet,
        "mw_total": mw_total,
        "mw_served": mw_served,
        "mw_missed": mw_missed,
        "mw_service_rate": mw_service_rate,
        "mw_urgent_total": len(mw_urgent),
        "mw_urgent_served": mw_urgent_served,
        "mw_urgent_rate": mw_urgent_rate,
        "mw_normal_total": len(mw_normal),
        "mw_normal_served": mw_normal_served,
        "mw_normal_rate": mw_normal_rate,
        "mw_n_slow": mw_n_slow,
        "mw_n_fast": mw_n_fast,
        "mw_n_uav": mw_n_uav,
        "mw_fleet_used": mw_n_slow + mw_n_fast + mw_n_uav,
    }
    return slot_data, fleet, problem, cfg, stats


def main():
    output_dir = Path("comparison_gi_alns")
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = {
        "gi_only": "Greedy Insertion (baseline)",
        "gi_alns": "Greedy Insertion + Improved ALNS",
    }

    all_stats = {}

    for strat_key, strat_name in strategies.items():
        print(f"\n{'='*80}", flush=True)
        print(f"  Strategy: {strat_name}", flush=True)
        print(f"{'='*80}", flush=True)

        if strat_key == "gi_only":
            cfg = Config(
                CONSTRUCTION_STRATEGY="deadline",
                PLOT_MAX_SLOTS=10,
                T_TOTAL=150,
                MEASUREMENT_START=0,
                MEASUREMENT_END=150,
            )
        else:
            cfg = Config(
                CONSTRUCTION_STRATEGY="alns",
                PLOT_MAX_SLOTS=10,
                T_TOTAL=150,
                MEASUREMENT_START=0,
                MEASUREMENT_END=150,
                ALNS_MAX_ITERATIONS=5000,
                ALNS_LOOKAHEAD_SLOTS=4,
                SA_INITIAL_TEMP=3000.0,
                SA_COOLING_RATE=0.9998,
                ALNS_MAX_COVERAGE_LOSS=2,
                BETA_WAITING=10.0,
            )

        slot_data, fleet, problem, cfg, stats = run_with_strategy(strat_key, cfg)
        all_stats[strat_key] = stats

        # Output subfolder
        strat_dir = output_dir / strat_key
        strat_dir.mkdir(parents=True, exist_ok=True)

        export_requests_csv(slot_data, strat_dir / "requests.csv")
        plot_per_slot_mcs_routes(slot_data, cfg, problem, strat_dir)

        # Print summary
        print(f"\n  --- {strat_name} ---")
        print(f"  MW Service Rate:  {stats['mw_service_rate']:.2%}")
        print(f"  MW Urgent Rate:   {stats['mw_urgent_rate']:.2%} "
              f"({stats['mw_urgent_served']}/{stats['mw_urgent_total']})")
        print(f"  MW Normal Rate:   {stats['mw_normal_rate']:.2%} "
              f"({stats['mw_normal_served']}/{stats['mw_normal_total']})")
        print(f"  MW Fleet Used:    {stats['mw_fleet_used']} "
              f"(Slow={stats['mw_n_slow']}, Fast={stats['mw_n_fast']}, UAV={stats['mw_n_uav']})")
        print(f"  Total Generated:  {stats['total_generated']}")
        print(f"  Total Served:     {stats['total_served']}")
        print(f"  Total Missed:     {stats['total_missed']}")

    # Summary comparison
    gi = all_stats["gi_only"]
    alns = all_stats["gi_alns"]

    summary_lines = [
        "=" * 60,
        "GI-only vs GI+ALNS Comparison Summary",
        "=" * 60,
        "",
        f"{'Metric':<25} {'GI-only':>12} {'GI+ALNS':>12} {'Delta':>10}",
        f"{'-'*25} {'-'*12} {'-'*12} {'-'*10}",
        f"{'MW Service Rate':<25} {gi['mw_service_rate']:>11.2%} {alns['mw_service_rate']:>11.2%} "
        f"{alns['mw_service_rate'] - gi['mw_service_rate']:>+9.2%}",
        f"{'MW Urgent Rate':<25} {gi['mw_urgent_rate']:>11.2%} {alns['mw_urgent_rate']:>11.2%} "
        f"{alns['mw_urgent_rate'] - gi['mw_urgent_rate']:>+9.2%}",
        f"{'MW Normal Rate':<25} {gi['mw_normal_rate']:>11.2%} {alns['mw_normal_rate']:>11.2%} "
        f"{alns['mw_normal_rate'] - gi['mw_normal_rate']:>+9.2%}",
        f"{'MW Fleet Used':<25} {gi['mw_fleet_used']:>12d} {alns['mw_fleet_used']:>12d} "
        f"{alns['mw_fleet_used'] - gi['mw_fleet_used']:>+10d}",
        f"{'  Slow MCS':<25} {gi['mw_n_slow']:>12d} {alns['mw_n_slow']:>12d} "
        f"{alns['mw_n_slow'] - gi['mw_n_slow']:>+10d}",
        f"{'  Fast MCS':<25} {gi['mw_n_fast']:>12d} {alns['mw_n_fast']:>12d} "
        f"{alns['mw_n_fast'] - gi['mw_n_fast']:>+10d}",
        f"{'  UAV':<25} {gi['mw_n_uav']:>12d} {alns['mw_n_uav']:>12d} "
        f"{alns['mw_n_uav'] - gi['mw_n_uav']:>+10d}",
        f"{'Total Generated':<25} {gi['total_generated']:>12d} {alns['total_generated']:>12d}",
        f"{'Total Served':<25} {gi['total_served']:>12d} {alns['total_served']:>12d} "
        f"{alns['total_served'] - gi['total_served']:>+10d}",
        f"{'Total Missed':<25} {gi['total_missed']:>12d} {alns['total_missed']:>12d} "
        f"{alns['total_missed'] - gi['total_missed']:>+10d}",
        "",
        "ALNS Config: 5000 iterations, SA T0=3000, cooling=0.9998, lookahead=4 slots",
        "=" * 60,
    ]

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}")

    # Write summary file
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(f"\nOutput saved to: {output_dir.resolve()}")
    print(f"  - comparison_gi_alns/gi_only/   (route maps + CSV)")
    print(f"  - comparison_gi_alns/gi_alns/   (route maps + CSV)")
    print(f"  - comparison_gi_alns/summary.txt")


if __name__ == "__main__":
    main()
