import numpy as np
from pathlib import Path
from typing import List, Dict
from copy import deepcopy

from config import Config
from models import Node, VehicleState, Route, Solution
from problem import ChargingSchedulingProblem
from simulation import initialize_fleet, simulate_dispatch_window
from alns import ALNSSolver

__all__ = [
    "Config", "Node", "VehicleState", "Route", "Solution",
    "ChargingSchedulingProblem", "initialize_fleet", "simulate_dispatch_window",
    "run_simulation",
]


def run_simulation(cfg: Config = None, verbose: bool = True):
    """
    執行 rolling-horizon 批次調度，回傳逐 slot 資料。

    Args:
        cfg: 實驗配置
        verbose: 是否印出逐 slot 詳細資訊 (grid search 時設 False)

    Returns:
        slot_data: 每個 slot 的詳細資料 (requests, solution snapshot, metrics)
        fleet: 最終車隊狀態
        problem: ChargingSchedulingProblem 實例
        cfg: 配置
        stats: 最終統計 dict (含測量窗口服務率)
    """
    cfg = cfg or Config()
    problem = ChargingSchedulingProblem(cfg)
    if verbose:
        problem.print_config()

    # 初始化車隊
    fleet = initialize_fleet(cfg, problem.depot)
    active_count = sum(1 for vs in fleet if vs.is_active)
    reserve_count = sum(1 for vs in fleet if not vs.is_active)
    if verbose:
        print(f"Fleet initialized: active={active_count}, reserve={reserve_count}")

    # 請求池
    all_requests: List[Node] = []  # 所有已生成的請求 (歷史紀錄)
    backlog: List[Node] = []       # 上一 slot 未服務的 backlog

    # 統計
    total_generated = 0
    total_served = 0
    total_missed = 0
    slot_data: List[dict] = []

    num_slots = cfg.T_TOTAL // cfg.DELTA_T

    if verbose:
        print(f"\n{'='*80}")
        print(f"Batch Dispatching | {num_slots} slots x {cfg.DELTA_T}min = {cfg.T_TOTAL}min")
        print(f"Measurement Window: t={cfg.MEASUREMENT_START}-{cfg.MEASUREMENT_END} "
              f"(slot {cfg.MEASUREMENT_START // cfg.DELTA_T + 1}-{cfg.MEASUREMENT_END // cfg.DELTA_T})")
        print(f"{'='*80}\n")

    for k in range(1, num_slots + 1):
        slot_start = (k - 1) * cfg.DELTA_T
        slot_end = k * cfg.DELTA_T
        dispatch_time = slot_end
        next_decision_time = slot_end + cfg.DELTA_T

        # Step 1: 生成本期新請求
        new_requests = problem.generate_requests(slot_start, slot_number=k)
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
            if verbose:
                r_t = cfg.get_arrival_rate_profile(slot_start)
                active_vehicles = sum(1 for vs in fleet if vs.is_active)
                print(f"  Slot {k:3d} [t={slot_start:.0f}-{slot_end:.0f}] r(t)={r_t:.1f} | "
                      f"new=0, backlog=0, pool=0 | "
                      f"served={completed_committed}, missed={slot_missed} | "
                      f"fleet={active_vehicles}")
            slot_data.append({
                "slot": k, "slot_start": slot_start, "slot_end": slot_end,
                "new_requests": [], "active_backlog": [],
                "active_pool": [], "solution": None,
                "n_assigned": 0, "slot_served": completed_committed,
                "slot_missed": slot_missed, "backlog_out": [],
            })
            continue

        # Step 3b: 更新距離矩陣 (加入新節點)
        all_customer_nodes = [n for n in all_requests if n.status not in ('served', 'missed')]
        all_customer_nodes.extend(new_requests)
        all_requests.extend(new_requests)
        problem.setup_nodes(all_customer_nodes)

        # Step 4: 計算 vehicle release states

        # Step 5: Construction heuristic
        if cfg.CONSTRUCTION_STRATEGY == "regret2":
            solution = problem.regret2_insertion_construction(
                active_pool, fleet, dispatch_window_end=next_decision_time
            )
        else:
            solution = problem.greedy_insertion_construction(
                active_pool, fleet, dispatch_window_end=next_decision_time
            )

        # Step 5.5: ALNS improvement
        assigned_count = sum(len(r.nodes) for r in solution.get_all_routes())
        if assigned_count > 0 and getattr(cfg, 'ALNS_MAX_ITERATIONS', 0) > 0:
            solver = ALNSSolver(problem, cfg)
            solution = solver.solve(solution, dispatch_window_end=next_decision_time)
            if getattr(cfg, 'ENABLE_CROSS_FLEET_LS', False):
                solution = solver.cross_fleet_local_search(solution)

        n_assigned = sum(1 for n in active_pool if n.status == 'assigned')
        n_unassigned = len(solution.unassigned_nodes)

        # Snapshot before simulation mutates node statuses
        snapshot = deepcopy(solution)

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

        # Step 7b: UAV re-queue — UAV 服務後的剩餘需求重新排入 MCS
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
                            uav_served=True,  # 防止 UAV 再次服務
                            residual_demand=node.residual_demand,
                            origin_slot=k,
                        )
                        problem._next_node_id += 1
                        backlog.append(followup)
                        total_generated += 1

        # 計算本 slot 完成的服務: simulate 中直接 served 的
        direct_served = 0
        for route in solution.get_all_routes():
            for node in route.nodes:
                if node.status == 'served':
                    direct_served += 1
        slot_served = completed_committed + direct_served
        total_served += direct_served

        if verbose:
            r_t = cfg.get_arrival_rate_profile(slot_start)
            active_vehicles = sum(1 for vs in fleet if vs.is_active)
            print(f"  Slot {k:3d} [t={slot_start:.0f}-{slot_end:.0f}] r(t)={r_t:.1f} | "
                  f"new={len(new_requests)}, backlog_in={len(active_backlog)}, "
                  f"pool={len(active_pool)} | "
                  f"assigned={n_assigned}, served={slot_served}, "
                  f"missed={slot_missed}, backlog_out={len(new_backlog)} | "
                  f"fleet={active_vehicles}")

        slot_data.append({
            "slot": k, "slot_start": slot_start, "slot_end": slot_end,
            "new_requests": new_requests, "active_backlog": active_backlog,
            "active_pool": active_pool, "solution": snapshot,
            "n_assigned": n_assigned, "slot_served": slot_served,
            "slot_missed": slot_missed, "backlog_out": new_backlog,
        })

    # ==================== Terminal: sweep remaining committed nodes ====================
    terminal_committed = 0
    for vs in fleet:
        for node in vs.committed_nodes:
            node.status = 'served'
            terminal_committed += 1
        vs.committed_nodes = []
        vs.committed_departures = []
    total_served += terminal_committed

    # Terminal: 標記所有 backlog 為 missed
    for req in backlog:
        req.status = 'missed'
        total_missed += 1
    backlog_remaining = len(backlog)
    backlog = []

    # ==================== 測量窗口服務率 ====================
    mw_requests = [r for r in all_requests
                   if cfg.MEASUREMENT_START <= r.ready_time < cfg.MEASUREMENT_END]
    mw_total = len(mw_requests)
    mw_served = sum(1 for r in mw_requests if r.status == 'served')
    mw_missed = sum(1 for r in mw_requests if r.status == 'missed')
    mw_service_rate = mw_served / mw_total if mw_total > 0 else 1.0

    # 分 urgent / normal
    mw_urgent = [r for r in mw_requests if r.node_type == 'urgent']
    mw_normal = [r for r in mw_requests if r.node_type == 'normal']
    mw_urgent_served = sum(1 for r in mw_urgent if r.status == 'served')
    mw_normal_served = sum(1 for r in mw_normal if r.status == 'served')
    mw_urgent_rate = mw_urgent_served / len(mw_urgent) if mw_urgent else 1.0
    mw_normal_rate = mw_normal_served / len(mw_normal) if mw_normal else 1.0

    # ==================== 測量窗口車輛使用統計 ====================
    # 統計在 MW 內實際服務過請求的 distinct vehicle IDs (per type)
    mw_served_requests = [r for r in mw_requests if r.status == 'served']
    mw_vehicles_used = {}  # vehicle_type -> set of vehicle_ids
    for r in mw_served_requests:
        vtype = r.served_by_vehicle_type
        vid = r.served_by_vehicle_id
        if vtype and vid >= 0:
            mw_vehicles_used.setdefault(vtype, set()).add(vid)

    mw_n_slow = len(mw_vehicles_used.get('mcs_slow', set()))
    mw_n_fast = len(mw_vehicles_used.get('mcs_fast', set()))
    mw_n_uav = len(mw_vehicles_used.get('uav', set()))

    if verbose:
        print(f"\n{'='*80}")
        print(f"Measurement Window [t={cfg.MEASUREMENT_START}-{cfg.MEASUREMENT_END}]")
        print(f"  Total requests: {mw_total}")
        print(f"  Served: {mw_served} | Missed: {mw_missed}")
        print(f"  Service Rate: {mw_service_rate:.2%}")
        print(f"  Urgent Rate:  {mw_urgent_rate:.2%} ({mw_urgent_served}/{len(mw_urgent)})")
        print(f"  Normal Rate:  {mw_normal_rate:.2%} ({mw_normal_served}/{len(mw_normal)})")
        print(f"  Vehicles Used: Slow={mw_n_slow}, Fast={mw_n_fast}, UAV={mw_n_uav} "
              f"(Total={mw_n_slow + mw_n_fast + mw_n_uav})")
        print(f"{'='*80}")

    stats = {
        "total_generated": total_generated,
        "total_served": total_served,
        "total_missed": total_missed,
        "backlog_remaining": backlog_remaining,
        "terminal_committed": terminal_committed,
        "all_requests": all_requests,
        "fleet": fleet,
        # 測量窗口統計
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
        # 車輛使用量
        "mw_n_slow": mw_n_slow,
        "mw_n_fast": mw_n_fast,
        "mw_n_uav": mw_n_uav,
        "mw_fleet_used": mw_n_slow + mw_n_fast + mw_n_uav,
    }
    return slot_data, fleet, problem, cfg, stats


def main():
    cfg = Config()
    slot_data, fleet, problem, cfg, stats = run_simulation(cfg)

    # Visualization & Report
    from visualization import (
        export_requests_csv, plot_per_slot_distribution,
        plot_per_slot_mcs_routes, print_terminal_report,
    )

    output_dir = Path("visualization_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    export_requests_csv(slot_data, output_dir / "requests.csv")
    plot_per_slot_distribution(slot_data, cfg, output_dir)
    plot_per_slot_mcs_routes(slot_data, cfg, problem, output_dir)
    print_terminal_report(slot_data, cfg, stats)


if __name__ == "__main__":
    main()
