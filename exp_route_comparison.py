"""
NN vs Greedy vs ALNS 路徑圖 + 效能指標比較實驗
=====================================================
單次批次調度 (無 Rolling Horizon)
車隊: 3 MCS-SLOW + 2 MCS-FAST + 1 UAV
需求: 10, 20, 30, 40, 50, 60, 70, 80

演算法:
  1. NN     — Nearest-Neighbor Chain
  2. Greedy — Greedy Insertion (baseline)
  3. ALNS   — Greedy 初始解 → ALNS 5000 iter
"""

import numpy as np
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from problem import ChargingSchedulingProblem
from alns import ALNSSolver
from simulation import initialize_fleet

# ================================================================
#  演算法顏色
# ================================================================
ALGO_COLORS = {
    'Greedy':   '#E53935',
    'NN':       '#FF9800',
    'ALNS':     '#1E88E5',
}
ALGO_MARKERS = {
    'Greedy':   'o',
    'NN':       '^',
    'ALNS':     's',
}


# ================================================================
#  指標提取
# ================================================================
def get_metrics(sol):
    active = [r for r in sol.get_all_routes() if len(r.nodes) > 0]
    served = sum(len(r.nodes) for r in sol.get_all_routes())
    missed = len(sol.unassigned_nodes)
    total = served + missed
    coverage = served / total * 100 if total > 0 else 100.0
    last_dep = max(r.departure_times[-1] for r in active if r.departure_times) if active else 0.0

    slow_distance = sum(r.total_distance for r in sol.mcs_routes
                        if r.vehicle_type == 'mcs_slow')
    fast_distance = sum(r.total_distance for r in sol.mcs_routes
                        if r.vehicle_type == 'mcs_fast')
    mcs_distance = slow_distance + fast_distance
    uav_distance = sum(r.total_distance for r in sol.uav_routes)
    mcs_served = sum(len(r.nodes) for r in sol.mcs_routes)
    uav_served = sum(len(r.nodes) for r in sol.uav_routes)

    return {
        'served': served, 'missed': missed, 'total': total,
        'coverage': coverage, 'last_dep': last_dep,
        'cost': sol.total_cost, 'distance': sol.total_distance,
        'slow_distance': slow_distance, 'fast_distance': fast_distance,
        'mcs_distance': mcs_distance, 'uav_distance': uav_distance,
        'mcs_served': mcs_served, 'uav_served': uav_served,
    }


# ================================================================
#  單次實驗
# ================================================================
def run_single_experiment(n_demand, seed=42):
    # 注意: 此實驗直接呼叫 greedy_insertion_construction / nearest_neighbor_construction，
    # 因此 CONSTRUCTION_STRATEGY 僅作為 greedy 的排序鍵；保留 "regret2" 以維持原本
    # "due_date 唯一鍵" 行為 (與舊版 fallback 排序完全一致)。
    cfg = Config(
        RANDOM_SEED=seed,
        N_REQUESTS=n_demand,
        CONSTRUCTION_STRATEGY="regret2",
        NUM_MCS_SLOW=3, NUM_MCS_FAST=2, NUM_UAV=1,
        MCS_UNLIMITED_ENERGY=True, ALNS_MAX_ITERATIONS=5000,
    )

    problem = ChargingSchedulingProblem(cfg)
    fleet = initialize_fleet(cfg, problem.depot)
    requests = problem.generate_requests()
    problem.setup_nodes(requests)
    dwe = cfg.DELTA_T

    def reset():
        random.seed(seed)
        np.random.seed(seed)
        for n in requests:
            n.status = 'new'

    # 1. Greedy
    reset()
    greedy_sol = problem.greedy_insertion_construction(list(requests), fleet, dwe)

    # 2. NN
    reset()
    nn_sol = problem.nearest_neighbor_construction(list(requests), fleet, dwe)

    # 3. ALNS (Greedy 初始解)
    reset()
    greedy_for_alns = problem.greedy_insertion_construction(list(requests), fleet, dwe)
    alns_sol = ALNSSolver(problem, cfg).solve(greedy_for_alns, dispatch_window_end=dwe)

    return {
        'NN': nn_sol,
        'Greedy': greedy_sol,
        'ALNS': alns_sol,
    }, problem


# ================================================================
#  主程式
# ================================================================
def main():
    demand_levels = [10, 20, 30, 40, 50, 60, 70, 80]
    target_n_for_curve = 60
    seed = 42
    out_dir = Path("output/route_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    algo_names = ['NN', 'Greedy', 'ALNS']

    print("=" * 70)
    print("  NN vs Greedy vs ALNS")
    print("  Fleet: 3 SLOW + 2 FAST + 1 UAV")
    print("=" * 70)

    all_metrics = {name: [] for name in algo_names}
    solutions_at_target = None

    for n_demand in demand_levels:
        t0 = time.time()
        print(f"\n{'─'*50}")
        print(f"  Demand = {n_demand}")
        print(f"{'─'*50}")

        solutions, problem = run_single_experiment(n_demand, seed)

        if n_demand == target_n_for_curve:
            solutions_at_target = solutions

        for name in algo_names:
            m = get_metrics(solutions[name])
            all_metrics[name].append(m)
            print(f"  {name:>10s}: served={m['served']:>2}, missed={m['missed']:>2}, "
                  f"cov={m['coverage']:>5.1f}%, last_dep={m['last_dep']:>6.1f}min, "
                  f"cost={m['cost']:>8.0f}")

        elapsed = time.time() - t0
        print(f"  ({elapsed:.1f}s)")

    # ==============================================================
    #  圖 A: Coverage Rate vs N
    # ==============================================================
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in algo_names:
        cov = [m['coverage'] for m in all_metrics[name]]
        ax.plot(demand_levels, cov,
                marker=ALGO_MARKERS[name], color=ALGO_COLORS[name],
                linewidth=2, markersize=8, label=name, zorder=5)

    ax.set_xlabel('Number of Requests (N)', fontsize=12)
    ax.set_ylabel('Coverage Rate (%)', fontsize=12)
    ax.set_title('Coverage Rate vs Demand\n[Fleet: 3 SLOW + 2 FAST + 1 UAV]',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(demand_levels)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "fig_coverage_vs_demand.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [Saved] {path}")

    # ==============================================================
    #  圖 B: Last Completion Time vs N
    # ==============================================================
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in algo_names:
        dep = [m['last_dep'] for m in all_metrics[name]]
        ax.plot(demand_levels, dep,
                marker=ALGO_MARKERS[name], color=ALGO_COLORS[name],
                linewidth=2, markersize=8, label=name, zorder=5)

    ax.set_xlabel('Number of Requests (N)', fontsize=12)
    ax.set_ylabel('Last Completion Time (min)', fontsize=12)
    ax.set_title('Last Completion Time vs Demand\n'
                 '[Fleet: 3 SLOW + 2 FAST + 1 UAV]',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(demand_levels)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "fig_completion_time_vs_demand.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")

    # ==============================================================
    #  圖 B2: Served Count vs Last Completion Time (固定 N = 60)
    #  y 軸里程碑 = {10, 20, 30, 40, 50, 60}
    #  x 軸 = 該演算法服務到第 k 個客戶的時間
    #  每個演算法一條折線；upper-left = better
    # ==============================================================
    milestones = [10, 20, 30, 40, 50, 60]
    if solutions_at_target is None:
        print(f"  [Warning] No solutions captured at N={target_n_for_curve}; "
              f"skipping efficiency plot.")
    else:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for name in algo_names:
            sol = solutions_at_target[name]
            completion_times = []
            for r in sol.get_all_routes():
                completion_times.extend(r.departure_times)
            completion_times.sort()
            served = len(completion_times)

            xs, ys = [], []
            for k in milestones:
                if k <= served:
                    xs.append(completion_times[k - 1])
                    ys.append(k)
            # 加上實際終點 (若不重複於最後一個里程碑)
            if served > 0 and (not ys or ys[-1] != served):
                xs.append(completion_times[-1])
                ys.append(served)
            if not xs:
                continue

            ax.plot(xs, ys,
                    marker=ALGO_MARKERS[name], color=ALGO_COLORS[name],
                    linewidth=2, markersize=8,
                    label=f"{name} (served {served})", zorder=5)

        ax.set_xlabel('Last Completion Time (min)', fontsize=12)
        ax.set_ylabel('Served Customers', fontsize=12)
        ax.set_title(f'Throughput vs Completion Time  '
                     f'[N = {target_n_for_curve}, '
                     f'Fleet: 3 SLOW + 2 FAST + 1 UAV]\n'
                     f'Upper-left = better',
                     fontsize=13, fontweight='bold')
        ax.set_yticks(milestones)
        ax.set_ylim(milestones[0] - 5, milestones[-1] + 5)
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / "fig_served_vs_completion_time.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {path}")

    # ==============================================================
    #  彙總表
    # ==============================================================
    print(f"\n{'='*90}")
    print(f"  {'N':>3} | {'NN':^22s} | {'Greedy':^22s} | {'ALNS':^22s}")
    print(f"  {'':>3} | {'Cov%':>5} {'Srv':>3} {'Dep':>7} {'Cost':>7}"
          f" | {'Cov%':>5} {'Srv':>3} {'Dep':>7} {'Cost':>7}"
          f" | {'Cov%':>5} {'Srv':>3} {'Dep':>7} {'Cost':>7}")
    print(f"  {'-'*86}")
    for i, n in enumerate(demand_levels):
        parts = []
        for name in algo_names:
            m = all_metrics[name][i]
            parts.append(f"{m['coverage']:>4.0f}% {m['served']:>3} {m['last_dep']:>7.1f} {m['cost']:>7.0f}")
        print(f"  {n:>3} | {' | '.join(parts)}")
    print(f"  {'='*86}")
    print(f"\n  All figures saved to {out_dir}")


if __name__ == "__main__":
    main()
