"""
Experiment: Coverage / Last-Completion-Time vs Demand Level
============================================================
Static single-dispatch (NOT rolling-horizon).

Fixed fleet : 3 Slow-MCS + 2 Fast-MCS + 1 UAV
Strategies  : NN | Regret-2 | Regret-2 + ALNS  (ALNS uses Regret-2 as initial)
Demand grid : [10, 20, 30, 40, 50, 60, 70, 80]
Replications: 10 seeds

Outputs:
    results/nn_regret2_alns/
        nn_regret2_alns.csv
        coverage_vs_demand.png
        last_completion_time_vs_demand.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # 避免 Windows Tcl/Tk 缺失導致畫圖崩潰
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
import csv
import time

from config import Config
from problem import ChargingSchedulingProblem
from simulation import initialize_fleet
from alns import ALNSSolver

# ── Experiment settings ──────────────────────────────────────────
DEMAND_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80]
NUM_SEEDS = 10
OUTPUT_DIR = Path("results") / "nn_regret2_alns"

# Fixed fleet
NUM_MCS_SLOW = 3
NUM_MCS_FAST = 2
NUM_UAV = 1


def make_cfg(demand: int, seed: int) -> Config:
    """Create config for a single (demand, seed) instance."""
    return Config(
        RANDOM_SEED=seed,
        NUM_MCS_SLOW=NUM_MCS_SLOW,
        NUM_MCS_FAST=NUM_MCS_FAST,
        NUM_UAV=NUM_UAV,
        N_REQUESTS=demand,
        MCS_UNLIMITED_ENERGY=True,
        ALNS_MAX_ITERATIONS=1000,
        SA_COOLING_RATE=0.9954,       # T drops to ~1% at iter 1000
    )


def run_single(demand: int, seed: int) -> dict:
    """Run all 3 methods on a single (demand, seed) instance."""
    cfg = make_cfg(demand, seed)

    # Generate instance (single static dispatch)
    problem = ChargingSchedulingProblem(cfg)
    requests = problem.generate_requests()
    problem.setup_nodes(requests)
    fleet = initialize_fleet(cfg, problem.depot)

    out = {}

    # ── Method 1: Nearest-Neighbor ──
    req_n = deepcopy(requests)
    fl_n = deepcopy(fleet)
    sol_n = problem.nearest_neighbor_construction(req_n, fl_n)
    out['nn'] = _extract(sol_n)

    # ── Method 2: Regret-2 Insertion ──
    req_r = deepcopy(requests)
    fl_r = deepcopy(fleet)
    sol_r = problem.regret2_insertion_construction(req_r, fl_r)
    out['regret2'] = _extract(sol_r)

    # ── Method 3: Regret-2 + ALNS (Regret-2 used as initial solution) ──
    req_a = deepcopy(requests)
    fl_a = deepcopy(fleet)
    sol_init = problem.regret2_insertion_construction(req_a, fl_a)
    solver = ALNSSolver(problem, cfg)
    sol_alns = solver.solve(sol_init)
    out['regret2_alns'] = _extract(sol_alns)

    return out


def _last_completion_time(sol) -> float:
    """
    Makespan: 最後一台車完成最後一個服務的時刻
    = max over routes of (last departure time)
    """
    last_t = 0.0
    for r in sol.get_all_routes():
        if not r.nodes or not r.departure_times:
            continue
        t = r.departure_times[-1]
        if t > last_t:
            last_t = t
    return last_t


def _extract(sol) -> dict:
    """Extract key metrics from a Solution."""
    n_urgent_miss = sum(1 for n in sol.unassigned_nodes if n.node_type == 'urgent')
    n_normal_miss = sum(1 for n in sol.unassigned_nodes if n.node_type == 'normal')
    return {
        'coverage': sol.coverage_rate,
        'cost': sol.total_cost,
        'avg_wait': sol.avg_waiting_time,
        'avg_wait_urgent': sol.avg_waiting_time_urgent,
        'avg_wait_normal': sol.avg_waiting_time_normal,
        'route_time_sum': sol.total_time,
        'last_completion': _last_completion_time(sol),
        'unassigned': len(sol.unassigned_nodes),
        'urgent_miss': n_urgent_miss,
        'normal_miss': n_normal_miss,
    }


# ── Main ─────────────────────────────────────────────────────────
def run_experiment():
    OUTPUT_DIR.mkdir(exist_ok=True)
    t_start = time.time()

    methods = ['nn', 'regret2', 'regret2_alns']
    all_cov = {m: {d: [] for d in DEMAND_LEVELS} for m in methods}
    all_lct = {m: {d: [] for d in DEMAND_LEVELS} for m in methods}
    raw_rows = []

    print(f"Fleet: {NUM_MCS_SLOW} Slow + {NUM_MCS_FAST} Fast + {NUM_UAV} UAV | Seeds: {NUM_SEEDS}")
    print(f"Demands: {DEMAND_LEVELS}")
    print(f"Strategies: NN | Regret-2 | Regret-2 + ALNS")
    print(f"Output dir: {OUTPUT_DIR}/")
    print("=" * 80)

    for demand in DEMAND_LEVELS:
        for seed in range(NUM_SEEDS):
            t0 = time.time()
            res = run_single(demand, seed)
            elapsed = time.time() - t0

            for m in methods:
                all_cov[m][demand].append(res[m]['coverage'])
                all_lct[m][demand].append(res[m]['last_completion'])

            raw_rows.append({
                'demand': demand, 'seed': seed, **{
                    f'{m}_{k}': res[m][k]
                    for m in methods for k in res[m]
                }
            })

            print(f"  D={demand:>2}  seed={seed}  "
                  f"NN={res['nn']['coverage']:5.1%}(LCT={res['nn']['last_completion']:6.1f})  "
                  f"R2={res['regret2']['coverage']:5.1%}(LCT={res['regret2']['last_completion']:6.1f})  "
                  f"R2+ALNS={res['regret2_alns']['coverage']:5.1%}(LCT={res['regret2_alns']['last_completion']:6.1f})  "
                  f"({elapsed:.1f}s)")

        print(f"  >> D={demand} avg coverage:  "
              f"NN={np.mean(all_cov['nn'][demand]):.1%}  "
              f"R2={np.mean(all_cov['regret2'][demand]):.1%}  "
              f"R2+ALNS={np.mean(all_cov['regret2_alns'][demand]):.1%}\n")

    # ── Summary table ──
    print("=" * 80)
    print(f"{'Demand':>6} | {'NN cov':>9} {'R2 cov':>9} {'ALNS cov':>9} | "
          f"{'NN LCT':>9} {'R2 LCT':>9} {'ALNS LCT':>9}")
    print("-" * 80)
    for d in DEMAND_LEVELS:
        print(f"{d:>6} | "
              f"{np.mean(all_cov['nn'][d]):>8.1%} "
              f"{np.mean(all_cov['regret2'][d]):>8.1%} "
              f"{np.mean(all_cov['regret2_alns'][d]):>8.1%} | "
              f"{np.mean(all_lct['nn'][d]):>9.1f} "
              f"{np.mean(all_lct['regret2'][d]):>9.1f} "
              f"{np.mean(all_lct['regret2_alns'][d]):>9.1f}")
    print("=" * 80)
    print(f"Total wall time: {time.time() - t_start:.1f}s")

    # ── Save outputs ──
    plot_coverage(all_cov)
    plot_last_completion(all_lct)
    save_csv(raw_rows)


# ── Plot: Coverage ───────────────────────────────────────────────
def plot_coverage(results):
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = [
        ('nn',           'Nearest-Neighbor', '#7f8c8d', 's'),
        ('regret2',      'Regret-2',         '#3498db', '^'),
        ('regret2_alns', 'Regret-2 + ALNS',  '#e74c3c', 'o'),
    ]

    for method, label, color, marker in styles:
        means = [np.mean(results[method][d]) for d in DEMAND_LEVELS]
        stds  = [np.std(results[method][d])  for d in DEMAND_LEVELS]

        ax.plot(DEMAND_LEVELS, means, label=label, color=color,
                marker=marker, linewidth=2, markersize=8)
        ax.fill_between(DEMAND_LEVELS,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.15)

    ax.set_xlabel('Number of Requests', fontsize=13)
    ax.set_ylabel('Coverage Rate', fontsize=13)
    ax.set_title(
        f'Coverage Rate vs Demand Level\n'
        f'(Fleet: {NUM_MCS_SLOW} Slow + {NUM_MCS_FAST} Fast + {NUM_UAV} UAV)',
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(DEMAND_LEVELS)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / 'coverage_vs_demand.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {path}")


# ── Plot: Last Completion Time (Makespan) ────────────────────────
def plot_last_completion(results):
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = [
        ('nn',           'Nearest-Neighbor', '#7f8c8d', 's'),
        ('regret2',      'Regret-2',         '#3498db', '^'),
        ('regret2_alns', 'Regret-2 + ALNS',  '#e74c3c', 'o'),
    ]

    for method, label, color, marker in styles:
        means = [np.mean(results[method][d]) for d in DEMAND_LEVELS]
        stds  = [np.std(results[method][d])  for d in DEMAND_LEVELS]

        ax.plot(DEMAND_LEVELS, means, label=label, color=color,
                marker=marker, linewidth=2, markersize=8)
        ax.fill_between(DEMAND_LEVELS,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.15)

    ax.set_xlabel('Number of Requests', fontsize=13)
    ax.set_ylabel('Last Completion Time (min)', fontsize=13)
    ax.set_title(
        f'Last Completion Time vs Demand Level\n'
        f'(Fleet: {NUM_MCS_SLOW} Slow + {NUM_MCS_FAST} Fast + {NUM_UAV} UAV)',
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.set_xticks(DEMAND_LEVELS)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / 'last_completion_time_vs_demand.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved: {path}")


# ── CSV ──────────────────────────────────────────────────────────
def save_csv(rows):
    if not rows:
        return
    path = OUTPUT_DIR / 'nn_regret2_alns.csv'
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {path}")


if __name__ == '__main__':
    run_experiment()
