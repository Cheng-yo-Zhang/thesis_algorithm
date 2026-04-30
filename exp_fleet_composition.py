"""
Fleet Composition Sensitivity — methodology pre-experiment.
============================================================

Purpose
-------
Justify the (NUM_MCS_SLOW, NUM_MCS_FAST, NUM_UAV) = (3, 2, 1) baseline used
in the four main experiments via a 2-D sweep over (slow, fast).  UAV count
is fixed at 1 because exp_uav_vs_urgent_ratio.py and exp_uav_vs_tw.py both
show that the urgent miss-rate gain saturates at K = 1 across the relevant
range of (rho, urgent TW).

Fixed
-----
    N_REQUESTS    = 30           (representative working point)
    URGENT_RATIO  = 0.5
    NUM_UAV       = 1
    construction  = regret2 + ALNS (5000 iter)
    seeds         = 42..51       (10 replications)

Sweep
-----
    NUM_MCS_SLOW  in {1, 2, 3, 4, 5}
    NUM_MCS_FAST  in {1, 2, 3, 4, 5}
    => 25 cells x 10 seeds = 250 runs

Selection criterion
-------------------
    Smallest capital-cost fleet (slow, fast, uav) satisfying simultaneously:
        overall service rate >= 95%
        urgent  service rate >= 95%
    where capital_cost = COST_SLOW*Ns + COST_FAST*Nf + COST_UAV*Nu.

Output
------
    results/fleet_composition/
        raw.csv                        per-run rows
        summary.csv                    per-(slow, fast) aggregates
        fig_heatmap_service_rate.png   overall service rate
        fig_heatmap_urgent_rate.png    urgent service rate
        fig_heatmap_cost.png           total objective Z (lower is better)
        fig_pareto.png                 capital cost vs service rate
"""

import csv
import time
import random
from copy import deepcopy
from math import sqrt
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from models import Node
from problem import ChargingSchedulingProblem
from simulation import initialize_fleet
from alns import ALNSSolver


# === Experiment Parameters ===
N_REQUESTS = 30
URGENT_RATIO = 0.5
NUM_UAV = 1
ALNS_ITER = 5000
NUM_REPLICATIONS = 10
BASE_SEED = 42

SLOW_GRID = [1, 2, 3, 4, 5]
FAST_GRID = [1, 2, 3, 4, 5]

# Selection thresholds
SERVICE_THRESHOLD = 0.95
URGENT_THRESHOLD = 0.95

# Capital cost proxy (relative units; documented in paper §V-A)
COST_SLOW = 1.0
COST_FAST = 1.5
COST_UAV = 0.5

OUTPUT_DIR = Path("results") / "fleet_composition"


# ================================================================
#  Helpers (self-contained: do not import from other exp_* files)
# ================================================================
def _generate_static_requests(problem: ChargingSchedulingProblem,
                              n: int, cfg: Config) -> list:
    """Generate n one-shot static requests (mirrors exp_fleet_vs_demand)."""
    nodes = []
    for _ in range(n):
        node_id = problem._next_node_id
        problem._next_node_id += 1

        is_urgent = np.random.random() < cfg.URGENT_RATIO

        if (is_urgent and cfg.URGENT_CORNER_RATIO > 0
                and np.random.random() < cfg.URGENT_CORNER_RATIO):
            corner = random.choice(cfg.CORNER_POSITIONS)
            x = np.clip(np.random.normal(corner[0], cfg.URGENT_CORNER_SIGMA),
                        0, cfg.AREA_SIZE)
            y = np.clip(np.random.normal(corner[1], cfg.URGENT_CORNER_SIGMA),
                        0, cfg.AREA_SIZE)
        else:
            x = np.random.uniform(0, cfg.AREA_SIZE)
            y = np.random.uniform(0, cfg.AREA_SIZE)

        if is_urgent:
            node_type = 'urgent'
            demand = problem._sample_truncated_demand(
                cfg.URGENT_ENERGY_MIN, cfg.URGENT_ENERGY_MAX)
            tw_width = np.random.uniform(cfg.URGENT_TW_MIN, cfg.URGENT_TW_MAX)
            ev_soc = np.random.uniform(cfg.URGENT_SOC_MIN, cfg.URGENT_SOC_MAX)
        else:
            node_type = 'normal'
            demand = problem._sample_truncated_demand(
                cfg.NORMAL_ENERGY_MIN, cfg.NORMAL_ENERGY_MAX)
            tw_width = np.random.uniform(cfg.NORMAL_TW_MIN, cfg.NORMAL_TW_MAX)
            ev_soc = max(0.05, 1.0 - demand / cfg.EV_BATTERY_CAPACITY)

        t_i = np.random.uniform(0, cfg.DELTA_T)
        d_i = t_i + tw_width

        nodes.append(Node(
            id=node_id, x=x, y=y, demand=demand,
            ready_time=t_i, due_date=d_i, service_time=0,
            node_type=node_type, status='new',
            ev_current_soc=ev_soc,
        ))
    return nodes


def _solve_once(requests, fleet, problem, cfg) -> 'Solution':
    sol = problem.regret2_insertion_construction(requests, fleet)
    if cfg.ALNS_MAX_ITERATIONS > 0:
        sol = ALNSSolver(problem, cfg).solve(sol)
    return sol


# ================================================================
#  Single run
# ================================================================
def run_one(num_slow: int, num_fast: int, num_uav: int, seed: int) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    cfg = Config(
        RANDOM_SEED=seed,
        N_REQUESTS=N_REQUESTS,
        URGENT_RATIO=URGENT_RATIO,
        NUM_MCS_SLOW=num_slow,
        NUM_MCS_FAST=num_fast,
        NUM_UAV=num_uav,
        CONSTRUCTION_STRATEGY="regret2",
        ALNS_MAX_ITERATIONS=ALNS_ITER,
    )
    problem = ChargingSchedulingProblem(cfg)
    fleet = initialize_fleet(cfg, problem.depot)
    requests = _generate_static_requests(problem, N_REQUESTS, cfg)
    problem.setup_nodes(requests)

    t0 = time.time()
    sol = _solve_once(deepcopy(requests), deepcopy(fleet), problem, cfg)
    elapsed = time.time() - t0

    sol.calculate_total_cost(total_customers=N_REQUESTS)

    n_total = len(requests)
    n_urgent = sum(1 for r in requests if r.node_type == 'urgent')
    n_normal = n_total - n_urgent
    n_assigned = sum(len(r.nodes) for r in sol.get_all_routes())
    n_urg_miss = sum(1 for n in sol.unassigned_nodes if n.node_type == 'urgent')
    n_nor_miss = sum(1 for n in sol.unassigned_nodes if n.node_type == 'normal')

    return {
        "num_slow": num_slow,
        "num_fast": num_fast,
        "num_uav": num_uav,
        "seed": seed,
        "n_requests": n_total,
        "n_assigned": n_assigned,
        "service_rate": n_assigned / n_total,
        "urgent_rate": (n_urgent - n_urg_miss) / n_urgent if n_urgent else 1.0,
        "normal_rate": (n_normal - n_nor_miss) / n_normal if n_normal else 1.0,
        "total_cost": sol.total_cost,
        "total_distance": sol.total_distance,
        "elapsed_sec": elapsed,
    }


# ================================================================
#  Sweep
# ================================================================
def run_sweep() -> list:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(SLOW_GRID) * len(FAST_GRID) * NUM_REPLICATIONS
    rows = []
    run_idx = 0
    t_start = time.time()

    print("=" * 80)
    print("Fleet Composition Sensitivity")
    print("=" * 80)
    print(f"  N            = {N_REQUESTS}")
    print(f"  rho          = {URGENT_RATIO}")
    print(f"  NUM_UAV      = {NUM_UAV}")
    print(f"  SLOW grid    = {SLOW_GRID}")
    print(f"  FAST grid    = {FAST_GRID}")
    print(f"  reps         = {NUM_REPLICATIONS}")
    print(f"  ALNS iter    = {ALNS_ITER}")
    print(f"  total runs   = {total_runs}")
    print("=" * 80)

    for ns in SLOW_GRID:
        for nf in FAST_GRID:
            cell_rows = []
            for rep in range(NUM_REPLICATIONS):
                run_idx += 1
                seed = BASE_SEED + rep
                row = run_one(ns, nf, NUM_UAV, seed)
                rows.append(row)
                cell_rows.append(row)

            elapsed = time.time() - t_start
            sr = np.mean([r['service_rate'] for r in cell_rows])
            ur = np.mean([r['urgent_rate'] for r in cell_rows])
            cost = np.mean([r['total_cost'] for r in cell_rows])
            print(f"  [{run_idx:3d}/{total_runs}] S={ns} F={nf} U={NUM_UAV} | "
                  f"sr={sr:.3f} ur={ur:.3f} cost={cost:7.0f} "
                  f"[{elapsed:.0f}s elapsed]")

    print("=" * 80)
    print(f"Sweep done in {time.time() - t_start:.1f}s")
    return rows


# ================================================================
#  Aggregate
# ================================================================
def _ci95(values):
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    h = 1.96 * s / sqrt(len(values)) if len(values) > 1 else 0.0
    return m, s, h


def aggregate(rows: list) -> list:
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r['num_slow'], r['num_fast']), []).append(r)

    summary = []
    for (ns, nf), cell in sorted(by_cell.items()):
        sr_m, _, sr_h = _ci95([r['service_rate'] for r in cell])
        ur_m, _, _ = _ci95([r['urgent_rate'] for r in cell])
        nr_m, _, _ = _ci95([r['normal_rate'] for r in cell])
        cost_m, _, _ = _ci95([r['total_cost'] for r in cell])
        capital = COST_SLOW * ns + COST_FAST * nf + COST_UAV * NUM_UAV
        summary.append({
            "num_slow": ns,
            "num_fast": nf,
            "num_uav": NUM_UAV,
            "capital_cost": capital,
            "mean_service_rate": sr_m,
            "ci95_service_rate": sr_h,
            "mean_urgent_rate": ur_m,
            "mean_normal_rate": nr_m,
            "mean_cost": cost_m,
        })
    return summary


def save_csv(rows: list, summary: list) -> None:
    raw_path = OUTPUT_DIR / "raw.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary_path = OUTPUT_DIR / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_path}")


# ================================================================
#  Plot
# ================================================================
def _make_grid(summary: list, key: str) -> np.ndarray:
    grid = np.zeros((len(SLOW_GRID), len(FAST_GRID)))
    for r in summary:
        i = SLOW_GRID.index(r['num_slow'])
        j = FAST_GRID.index(r['num_fast'])
        grid[i, j] = r[key]
    return grid


def _luminance_text_color(value: float, vmin: float, vmax: float,
                          cmap_name: str) -> str:
    cmap = plt.get_cmap(cmap_name)
    t = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    r, g, b, _ = cmap(t)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if luminance < 0.5 else 'black'


def plot_heatmap(grid: np.ndarray, title: str, fname: str,
                 fmt: str = "{:.2f}", cmap: str = "viridis",
                 contour_at: float = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, origin='lower', cmap=cmap, aspect='auto')
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(FAST_GRID)))
    ax.set_yticks(range(len(SLOW_GRID)))
    ax.set_xticklabels(FAST_GRID)
    ax.set_yticklabels(SLOW_GRID)
    ax.set_xlabel("Number of MCS-Fast", fontsize=12)
    ax.set_ylabel("Number of MCS-Slow", fontsize=12)
    ax.set_title(title, fontsize=12)

    vmin, vmax = float(grid.min()), float(grid.max())
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            color = _luminance_text_color(grid[i, j], vmin, vmax, cmap)
            ax.text(j, i, fmt.format(grid[i, j]),
                    ha='center', va='center', color=color, fontsize=9)

    # Highlight (3, 2)
    if 3 in SLOW_GRID and 2 in FAST_GRID:
        i = SLOW_GRID.index(3)
        j = FAST_GRID.index(2)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor='red', linewidth=2.5))

    if contour_at is not None:
        x = np.arange(grid.shape[1])
        y = np.arange(grid.shape[0])
        X, Y = np.meshgrid(x, y)
        cs = ax.contour(X, Y, grid, levels=[contour_at], colors='red',
                        linewidths=2, linestyles='--')
        ax.clabel(cs, inline=True, fontsize=9, fmt=lambda v: f"{v:.0f}")

    fig.tight_layout()
    out = OUTPUT_DIR / fname
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def plot_pareto(summary: list) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    capitals = np.array([r['capital_cost'] for r in summary])
    rates = np.array([r['mean_service_rate'] for r in summary])
    labels = [(r['num_slow'], r['num_fast']) for r in summary]

    sorted_idx = np.argsort(capitals)
    front = []
    best_rate = -1.0
    for i in sorted_idx:
        if rates[i] > best_rate:
            best_rate = rates[i]
            front.append(i)
    front = np.array(front)

    ax.scatter(capitals, rates * 100, s=55, alpha=0.55,
               color="#3498db", edgecolor='black', linewidth=0.5)
    ax.plot(capitals[front], rates[front] * 100, '-', color='#e74c3c',
            linewidth=2, label='Pareto front', zorder=4)

    selected_drawn = False
    for i, lbl in enumerate(labels):
        if lbl == (3, 2) and not selected_drawn:
            ax.scatter(capitals[i], rates[i] * 100, s=220,
                       facecolor='none', edgecolor='red', linewidth=2.5,
                       label='Selected (3,2,1)', zorder=5)
            selected_drawn = True
        ax.annotate(f"({lbl[0]},{lbl[1]})", (capitals[i], rates[i] * 100),
                    fontsize=7.5, alpha=0.7,
                    xytext=(4, 4), textcoords='offset points')

    ax.axhline(y=SERVICE_THRESHOLD * 100, color='gray',
               linestyle=':', linewidth=1.2,
               label=f'{SERVICE_THRESHOLD:.0%} service threshold')

    ax.set_xlabel(f"Capital Cost Proxy "
                  f"(slow={COST_SLOW}, fast={COST_FAST}, uav={COST_UAV})",
                  fontsize=11)
    ax.set_ylabel("Service Rate (%)", fontsize=11)
    ax.set_title(f"Fleet Composition — Pareto Front "
                 f"(N={N_REQUESTS}, rho={URGENT_RATIO}, UAV={NUM_UAV})",
                 fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig_pareto.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ================================================================
#  Selection report
# ================================================================
def select_and_report(summary: list) -> None:
    candidates = [r for r in summary
                  if r['mean_service_rate'] >= SERVICE_THRESHOLD
                  and r['mean_urgent_rate'] >= URGENT_THRESHOLD]
    print()
    print("=" * 80)
    print(f"Candidates with overall SR >= {SERVICE_THRESHOLD:.0%} "
          f"AND urgent SR >= {URGENT_THRESHOLD:.0%}:")
    print("=" * 80)

    if not candidates:
        print("  (none — relax thresholds or expand grid)")
        print("=" * 80)
        return

    candidates.sort(key=lambda r: (r['capital_cost'], -r['mean_service_rate']))
    print(f"  {'Slow':>4} {'Fast':>4} {'UAV':>4} "
          f"{'Capital':>8} {'SR':>6} {'UR':>6} {'NR':>6} {'Cost Z':>8}")
    print(f"  {'-'*4} {'-'*4} {'-'*4} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for c in candidates:
        marker = "  <-- min capital" if c == candidates[0] else ""
        print(f"  {c['num_slow']:4d} {c['num_fast']:4d} {c['num_uav']:4d} "
              f"{c['capital_cost']:8.2f} "
              f"{c['mean_service_rate']:6.3f} "
              f"{c['mean_urgent_rate']:6.3f} "
              f"{c['mean_normal_rate']:6.3f} "
              f"{c['mean_cost']:8.0f}{marker}")
    print("=" * 80)

    chosen = candidates[0]
    print(f"\nRecommended composition: "
          f"(slow={chosen['num_slow']}, fast={chosen['num_fast']}, "
          f"uav={chosen['num_uav']})")
    print(f"  service rate = {chosen['mean_service_rate']:.3f}, "
          f"urgent rate = {chosen['mean_urgent_rate']:.3f}, "
          f"capital = {chosen['capital_cost']:.2f}")

    # Compare against the (3, 2, 1) default
    target = next((r for r in summary
                   if r['num_slow'] == 3 and r['num_fast'] == 2), None)
    if target is not None:
        gap = chosen['mean_service_rate'] - target['mean_service_rate']
        print(f"\n(3, 2, 1) for reference: "
              f"SR={target['mean_service_rate']:.3f}, "
              f"UR={target['mean_urgent_rate']:.3f}, "
              f"capital={target['capital_cost']:.2f}, "
              f"gap vs recommended = {gap*100:+.2f} pp")


# ================================================================
#  Main
# ================================================================
def main():
    rows = run_sweep()
    summary = aggregate(rows)
    save_csv(rows, summary)

    sr_grid = _make_grid(summary, 'mean_service_rate') * 100
    ur_grid = _make_grid(summary, 'mean_urgent_rate') * 100
    cost_grid = _make_grid(summary, 'mean_cost')

    plot_heatmap(sr_grid,
                 f"(a) Overall Service Rate (%) — N={N_REQUESTS}, "
                 f"rho={URGENT_RATIO}, UAV={NUM_UAV}",
                 "fig_heatmap_service_rate.png", fmt="{:.1f}",
                 cmap="viridis", contour_at=SERVICE_THRESHOLD * 100)
    plot_heatmap(ur_grid,
                 f"(b) Urgent Service Rate (%)",
                 "fig_heatmap_urgent_rate.png", fmt="{:.1f}",
                 cmap="viridis", contour_at=URGENT_THRESHOLD * 100)
    plot_heatmap(cost_grid,
                 f"(c) Total Objective Z (lower is better)",
                 "fig_heatmap_cost.png", fmt="{:.0f}",
                 cmap="viridis_r")

    plot_pareto(summary)
    select_and_report(summary)


if __name__ == "__main__":
    main()
