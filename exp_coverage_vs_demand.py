"""
Experiment: Coverage Rate vs Demand Level
==========================================
Fixed fleet (3 Slow-MCS + 2 Fast-MCS + 1 UAV).
Compare: Greedy | Regret-2 | Regret-2 + ALNS

X-axis: number of requests [10, 20, 30, 40, 50, 60]
Y-axis: coverage rate (averaged over 10 seeds ± std)
"""

import numpy as np
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
DEMAND_LEVELS = [10, 20, 30, 40, 50, 60]
NUM_SEEDS = 10
OUTPUT_DIR = Path("results_coverage_vs_demand")

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

    # Generate instance
    problem = ChargingSchedulingProblem(cfg)
    requests = problem.generate_requests()
    problem.setup_nodes(requests)
    fleet = initialize_fleet(cfg, problem.depot)

    out = {}

    # ── Method 1: Greedy Insertion ──
    req_g = deepcopy(requests)
    fl_g = deepcopy(fleet)
    sol_g = problem.greedy_insertion_construction(req_g, fl_g)
    out['greedy'] = _extract(sol_g)

    # ── Method 2: Regret-2 Insertion ──
    req_r = deepcopy(requests)
    fl_r = deepcopy(fleet)
    sol_r = problem.regret2_insertion_construction(req_r, fl_r)
    out['regret2'] = _extract(sol_r)

    # ── Method 3: Regret-2 + ALNS ──
    req_a = deepcopy(requests)
    fl_a = deepcopy(fleet)
    sol_init = problem.regret2_insertion_construction(req_a, fl_a)
    solver = ALNSSolver(problem, cfg)
    sol_alns = solver.solve(sol_init)
    out['regret2_alns'] = _extract(sol_alns)

    return out


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
        'route_time': sol.total_time,
        'unassigned': len(sol.unassigned_nodes),
        'urgent_miss': n_urgent_miss,
        'normal_miss': n_normal_miss,
    }


# ── Main ─────────────────────────────────────────────────────────
def run_experiment():
    OUTPUT_DIR.mkdir(exist_ok=True)

    methods = ['greedy', 'regret2', 'regret2_alns']
    all_cov = {m: {d: [] for d in DEMAND_LEVELS} for m in methods}
    raw_rows = []

    print(f"Fleet: {NUM_MCS_SLOW} Slow + {NUM_MCS_FAST} Fast + {NUM_UAV} UAV | Seeds: {NUM_SEEDS}")
    print(f"Demands: {DEMAND_LEVELS}")
    print("=" * 72)

    for demand in DEMAND_LEVELS:
        for seed in range(NUM_SEEDS):
            t0 = time.time()
            res = run_single(demand, seed)
            elapsed = time.time() - t0

            for m in methods:
                all_cov[m][demand].append(res[m]['coverage'])

            raw_rows.append({
                'demand': demand, 'seed': seed, **{
                    f'{m}_{k}': res[m][k]
                    for m in methods for k in res[m]
                }
            })

            print(f"  D={demand:>2}  seed={seed}  "
                  f"Greedy={res['greedy']['coverage']:5.1%}  "
                  f"Regret-2={res['regret2']['coverage']:5.1%}  "
                  f"R2+ALNS={res['regret2_alns']['coverage']:5.1%}  "
                  f"({elapsed:.1f}s)")

        # Per-demand summary
        g = np.mean(all_cov['greedy'][demand])
        r = np.mean(all_cov['regret2'][demand])
        a = np.mean(all_cov['regret2_alns'][demand])
        print(f"  >> D={demand} avg:  Greedy={g:.1%}  Regret-2={r:.1%}  R2+ALNS={a:.1%}\n")

    # ── Summary table ──
    print("=" * 72)
    print(f"{'Demand':>8} | {'Greedy':>14} | {'Regret-2':>14} | {'R2+ALNS':>14}")
    print("-" * 72)
    for d in DEMAND_LEVELS:
        g = np.mean(all_cov['greedy'][d])
        r = np.mean(all_cov['regret2'][d])
        a = np.mean(all_cov['regret2_alns'][d])
        print(f"{d:>8} | {g:>13.1%} | {r:>13.1%} | {a:>13.1%}")
    print("=" * 72)

    # ── Save outputs ──
    plot_coverage(all_cov)
    save_csv(raw_rows)


# ── Plot ─────────────────────────────────────────────────────────
def plot_coverage(results):
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = [
        ('greedy',       'Greedy Insertion',  '#e74c3c', 'o'),
        ('regret2_alns', 'Regret-2 + ALNS',   '#2ecc71', '^'),
    ]

    for method, label, color, marker in styles:
        means = [np.mean(results[method][d]) for d in DEMAND_LEVELS]
        stds  = [np.std(results[method][d])  for d in DEMAND_LEVELS]

        ax.plot(DEMAND_LEVELS, means, label=label, color=color,
                marker=marker, linewidth=2, markersize=8)

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


# ── CSV ──────────────────────────────────────────────────────────
def save_csv(rows):
    if not rows:
        return
    path = OUTPUT_DIR / 'coverage_vs_demand.csv'
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {path}")


if __name__ == '__main__':
    run_experiment()
