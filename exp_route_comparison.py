"""
Greedy vs Sweep+R2 vs ALNS 路徑圖 + 效能指標比較實驗
=====================================================
單次批次調度 (無 Rolling Horizon)
車隊: 3 MCS-SLOW + 2 MCS-FAST + 1 UAV
需求: 10, 20, 30, 40, 50, 60, 70, 80

演算法:
  1. Greedy — Greedy Insertion (baseline)
  2. Sweep+R2 — Sweep 分區 + Regret-2
  3. ALNS(S) — Sweep+R2 初始解 → ALNS 5000 iter
  (4. ALNS(G) — 預留，可後續啟用)
"""

import numpy as np
import random
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import Config
from models import Node, Route, Solution
from problem import ChargingSchedulingProblem
from alns import ALNSSolver
from simulation import initialize_fleet

# ================================================================
#  車型顏色 & 演算法顏色
# ================================================================
TYPE_COLORS = {
    'mcs_slow': '#2196F3',
    'mcs_fast': '#FF9800',
    'uav':      '#4CAF50',
}
TYPE_LABELS = {
    'mcs_slow': 'MCS-SLOW',
    'mcs_fast': 'MCS-FAST',
    'uav':      'UAV',
}

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
#  路徑繪圖
# ================================================================
def plot_routes(ax, solution, problem, cfg, title):
    depot_x, depot_y = cfg.DEPOT_X, cfg.DEPOT_Y
    ax.scatter(depot_x, depot_y, c="black", marker="s", s=200,
               edgecolors="k", linewidths=1.2, zorder=10)
    ax.annotate("Depot", (depot_x, depot_y), fontsize=8, fontweight="bold",
                ha="center", va="bottom", xytext=(0, 8), textcoords="offset points")

    route_handles = []
    type_counter = {'mcs_slow': 0, 'mcs_fast': 0, 'uav': 0}

    for route in solution.mcs_routes + solution.uav_routes:
        if len(route.nodes) == 0:
            continue
        vtype = route.vehicle_type
        color = TYPE_COLORS[vtype]
        is_uav = (vtype == 'uav')
        ls = "--" if is_uav else "-"
        lw = 1.8 if is_uav else 2.0
        type_counter[vtype] += 1

        start = route.start_position if route.start_position else problem.depot
        xs = [start.x] + [n.x for n in route.nodes] + [depot_x]
        ys = [start.y] + [n.y for n in route.nodes] + [depot_y]
        ax.plot(xs, ys, ls, color=color, linewidth=lw, alpha=0.75, zorder=4)

        for seg in range(len(xs) - 1):
            mx, my = (xs[seg]+xs[seg+1])/2, (ys[seg]+ys[seg+1])/2
            ddx, ddy = xs[seg+1]-xs[seg], ys[seg+1]-ys[seg]
            if abs(ddx) > 0.01 or abs(ddy) > 0.01:
                ax.annotate("", xy=(mx+ddx*0.01, my+ddy*0.01), xytext=(mx, my),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

        for node in route.nodes:
            is_u = node.node_type == "urgent"
            ax.scatter(node.x, node.y,
                       c="#e74c3c" if is_u else "#3498db",
                       marker="*" if is_u else "o",
                       s=160 if is_u else 70,
                       edgecolors="k", linewidths=0.5, zorder=6)
            ax.annotate(f"EV{node.id}", (node.x, node.y), fontsize=5,
                        fontweight="bold", ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points")
        for seq_i, node in enumerate(route.nodes):
            ax.annotate(f"{seq_i+1}", (node.x, node.y), fontsize=5,
                        color=color, ha="left", va="top",
                        xytext=(4, -4), textcoords="offset points")

        label = f"{TYPE_LABELS[vtype]}-{type_counter[vtype]}: {route.get_node_ids()}"
        route_handles.append(Line2D([0], [0], color=color, linewidth=lw, linestyle=ls, label=label))

    if solution.unassigned_nodes:
        for node in solution.unassigned_nodes:
            ax.scatter(node.x, node.y, c="none", marker="o", s=120,
                       edgecolors="red", linewidths=2, zorder=7)
            ax.annotate(f"EV{node.id}\n(miss)", (node.x, node.y),
                        fontsize=5, color="red", ha="center", va="top",
                        xytext=(0, -8), textcoords="offset points")

    ax.set_xlim(-0.5, cfg.AREA_SIZE+0.5)
    ax.set_ylim(-0.5, cfg.AREA_SIZE+0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("X (km)", fontsize=9)
    ax.set_ylabel("Y (km)", fontsize=9)

    type_legend = [
        Line2D([0],[0], color=TYPE_COLORS['mcs_slow'], lw=2.5, label="MCS-SLOW"),
        Line2D([0],[0], color=TYPE_COLORS['mcs_fast'], lw=2.5, label="MCS-FAST"),
        Line2D([0],[0], color=TYPE_COLORS['uav'], lw=2.5, ls="--", label="UAV"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="#e74c3c",
               markeredgecolor="k", markersize=10, label="Urgent"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#3498db",
               markeredgecolor="k", markersize=7, label="Normal"),
    ]

    n_served = sum(len(r.nodes) for r in solution.get_all_routes())
    n_miss = len(solution.unassigned_nodes)
    ax.set_title(f"{title}\nServed={n_served}  Missed={n_miss}  "
                 f"Cost={solution.total_cost:.0f}  Dist={solution.total_distance:.1f}km",
                 fontsize=9)

    leg1 = ax.legend(handles=type_legend, fontsize=6, loc="upper left", framealpha=0.85)
    ax.add_artist(leg1)
    if route_handles:
        ax.legend(handles=route_handles, fontsize=5, loc="upper right", ncol=1, framealpha=0.8)


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
    return {
        'served': served, 'missed': missed, 'total': total,
        'coverage': coverage, 'last_dep': last_dep,
        'cost': sol.total_cost, 'distance': sol.total_distance,
    }


# ================================================================
#  單次實驗
# ================================================================
def run_single_experiment(n_demand, seed=42):
    cfg = Config(
        RANDOM_SEED=seed, USE_FIXED_DEMAND=True,
        FIXED_DEMAND_PER_SLOT=n_demand,
        CONSTRUCTION_STRATEGY="regret2",
        NUM_MCS_SLOW=3, NUM_MCS_FAST=2, NUM_UAV=1,
        MCS_UNLIMITED_ENERGY=True, ALNS_MAX_ITERATIONS=5000, T_TOTAL=2880,
    )

    problem = ChargingSchedulingProblem(cfg)
    fleet = initialize_fleet(cfg, problem.depot)
    requests = problem.generate_requests(slot_start=0, slot_number=1)
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
    seed = 42
    out_dir = Path("output/route_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    algo_names = ['NN', 'Greedy', 'ALNS']

    print("=" * 70)
    print("  NN vs Greedy vs ALNS")
    print("  Fleet: 3 SLOW + 2 FAST + 1 UAV")
    print("=" * 70)

    all_metrics = {name: [] for name in algo_names}

    for n_demand in demand_levels:
        t0 = time.time()
        print(f"\n{'─'*50}")
        print(f"  Demand = {n_demand}")
        print(f"{'─'*50}")

        solutions, problem = run_single_experiment(n_demand, seed)

        for name in algo_names:
            m = get_metrics(solutions[name])
            all_metrics[name].append(m)
            print(f"  {name:>10s}: served={m['served']:>2}, missed={m['missed']:>2}, "
                  f"cov={m['coverage']:>5.1f}%, last_dep={m['last_dep']:>6.1f}min, "
                  f"cost={m['cost']:>8.0f}")

        elapsed = time.time() - t0
        print(f"  ({elapsed:.1f}s)")

        # --- 路徑圖: 1×3 grid ---
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        for ax, name in zip(axes, algo_names):
            plot_routes(ax, solutions[name], problem, problem.cfg, f"{name} (N={n_demand})")

        fig.suptitle(f"Route Comparison — N={n_demand}  [3 SLOW + 2 FAST + 1 UAV]",
                     fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        path = out_dir / f"route_N{n_demand:02d}_3algo.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {path}")

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
        srv = [m['served'] for m in all_metrics[name]]
        ax.plot(demand_levels, dep,
                marker=ALGO_MARKERS[name], color=ALGO_COLORS[name],
                linewidth=2, markersize=8, label=name, zorder=5)
        for i, n in enumerate(demand_levels):
            ax.annotate(f'({srv[i]})', (n, dep[i]),
                        fontsize=7, color=ALGO_COLORS[name], fontweight='bold',
                        ha='left', va='bottom', xytext=(4, 3),
                        textcoords='offset points')

    ax.set_xlabel('Number of Requests (N)', fontsize=12)
    ax.set_ylabel('Last Completion Time (min)', fontsize=12)
    ax.set_title('Last Completion Time vs Demand\n'
                 '[Fleet: 3 SLOW + 2 FAST + 1 UAV]  (N) = served count',
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
    #  圖 C: Objective Value vs N
    # ==============================================================
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in algo_names:
        cost = [m['cost'] for m in all_metrics[name]]
        ax.plot(demand_levels, cost,
                marker=ALGO_MARKERS[name], color=ALGO_COLORS[name],
                linewidth=2, markersize=8, label=name, zorder=5)

    ax.set_xlabel('Number of Requests (N)', fontsize=12)
    ax.set_ylabel('Objective Value (Z)', fontsize=12)
    ax.set_title('Objective Value vs Demand\n[Fleet: 3 SLOW + 2 FAST + 1 UAV]',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(demand_levels)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "fig_objective_vs_demand.png"
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
