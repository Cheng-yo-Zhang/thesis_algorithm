"""
Algorithm Comparison — Service Rate vs Number of Requests (Static Single-Batch)
================================================================================
x-axis : N (number of requests, single batch)
y-axis : Service Rate (% of requests successfully assigned)
Lines  : Greedy (EDF) | Regret-2 | Regret-2 + ALNS

Fleet (fixed) : 5 Slow-MCS + 5 Fast-MCS + 5 UAV
Replications  : 10 seeds per (N, strategy)
"""

import csv
import json
import time
from copy import deepcopy
from math import sqrt
from pathlib import Path

import numpy as np

from alns import ALNSSolver
from config import Config
from problem import ChargingSchedulingProblem
from simulation import initialize_fleet


# === Experiment Parameters ===
N_VALUES = [10, 20, 30, 40, 50, 60, 70, 80]
NUM_REPLICATIONS = 10
BASE_SEED = 42
OUTPUT_DIR = Path("results_algorithm_comparison")

NUM_MCS_SLOW = 5
NUM_MCS_FAST = 5
NUM_UAV = 5

STRATEGIES = [
    {"name": "Greedy (EDF)",    "construction": "edf",     "alns_iter": 0},
    {"name": "Regret-2",        "construction": "regret2", "alns_iter": 0},
    {"name": "Regret-2 + ALNS", "construction": "regret2", "alns_iter": 5000},
]


def make_cfg(n_requests: int, strategy: dict, seed: int) -> Config:
    return Config(
        RANDOM_SEED=seed,
        N_REQUESTS=n_requests,
        NUM_MCS_SLOW=NUM_MCS_SLOW,
        NUM_MCS_FAST=NUM_MCS_FAST,
        NUM_UAV=NUM_UAV,
        MCS_UNLIMITED_ENERGY=True,
        CONSTRUCTION_STRATEGY=strategy["construction"],
        ALNS_MAX_ITERATIONS=strategy["alns_iter"],
    )


def solve_one(cfg: Config, strategy: dict) -> dict:
    problem = ChargingSchedulingProblem(cfg)
    fleet = initialize_fleet(cfg, problem.depot)
    requests = problem.generate_requests()
    problem.setup_nodes(requests)

    if strategy["construction"] == "regret2":
        sol = problem.regret2_insertion_construction(deepcopy(requests), deepcopy(fleet))
    else:
        sol = problem.greedy_insertion_construction(deepcopy(requests), deepcopy(fleet))

    if strategy["alns_iter"] > 0:
        solver = ALNSSolver(problem, cfg)
        sol = solver.solve(sol)

    n_total = len(requests)
    n_urgent = sum(1 for r in requests if r.node_type == 'urgent')
    n_normal = n_total - n_urgent
    n_assigned = sum(len(r.nodes) for r in sol.get_all_routes())
    n_missed = len(sol.unassigned_nodes)
    n_urgent_missed = sum(1 for n in sol.unassigned_nodes if n.node_type == 'urgent')
    n_normal_missed = sum(1 for n in sol.unassigned_nodes if n.node_type == 'normal')

    return {
        "service_rate": n_assigned / n_total if n_total > 0 else 1.0,
        "urgent_rate": (n_urgent - n_urgent_missed) / n_urgent if n_urgent > 0 else 1.0,
        "normal_rate": (n_normal - n_normal_missed) / n_normal if n_normal > 0 else 1.0,
        "n_assigned": n_assigned,
        "n_missed": n_missed,
        "cost": sol.total_cost,
    }


def ci95(arr):
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    h = 1.96 * s / sqrt(len(arr)) if len(arr) > 1 else 0.0
    return m, s, h


def run_experiment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    results = []

    total_runs = len(N_VALUES) * len(STRATEGIES) * NUM_REPLICATIONS
    run_count = 0

    print("=" * 80)
    print("Algorithm Comparison — Service Rate vs N (Static Single-Batch)")
    print("=" * 80)
    print(f"  N values: {N_VALUES}")
    print(f"  Strategies: {[s['name'] for s in STRATEGIES]}")
    print(f"  Replications: {NUM_REPLICATIONS}")
    print(f"  Total runs: {total_runs}")
    print()

    for n in N_VALUES:
        for strategy in STRATEGIES:
            rep_data = {"service_rate": [], "urgent_rate": [],
                        "normal_rate": [], "cost": []}

            for rep in range(NUM_REPLICATIONS):
                cfg = make_cfg(n, strategy, BASE_SEED + rep)
                metrics = solve_one(cfg, strategy)
                for key in rep_data:
                    rep_data[key].append(metrics[key])
                run_count += 1

            sr_m, sr_s, sr_h = ci95(rep_data["service_rate"])
            ur_m, _, _ = ci95(rep_data["urgent_rate"])
            nr_m, _, _ = ci95(rep_data["normal_rate"])
            cost_m, _, _ = ci95(rep_data["cost"])

            row = {
                "n_requests": n,
                "strategy": strategy["name"],
                "mean_service_rate": sr_m,
                "std_service_rate": sr_s,
                "ci95_half": sr_h,
                "mean_urgent_rate": ur_m,
                "mean_normal_rate": nr_m,
                "mean_cost": cost_m,
            }
            results.append(row)

            elapsed = time.time() - t_start
            print(f"  [{run_count:3d}/{total_runs}] N={n:3d} {strategy['name']:>17s} | "
                  f"rate={sr_m:.3f}±{sr_s:.3f} | "
                  f"urgent={ur_m:.3f} normal={nr_m:.3f} | "
                  f"cost={cost_m:7.0f} [{elapsed:.0f}s]")

    csv_path = OUTPUT_DIR / "algorithm_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_path = OUTPUT_DIR / "algorithm_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_time = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"Total time: {total_time / 60:.1f} min")
    print(f"Results saved to {csv_path}")

    plot_results(results)
    return results


def plot_results(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {
        "Greedy (EDF)": "#7f8c8d",
        "Regret-2": "#3498db",
        "Regret-2 + ALNS": "#e74c3c",
    }
    markers = {
        "Greedy (EDF)": "s",
        "Regret-2": "^",
        "Regret-2 + ALNS": "o",
    }

    for strategy_info in STRATEGIES:
        sname = strategy_info["name"]
        rows = [r for r in results if r["strategy"] == sname]
        ns = [r["n_requests"] for r in rows]
        rates = [r["mean_service_rate"] * 100 for r in rows]
        ci = [r["ci95_half"] * 100 for r in rows]
        urgent = [r["mean_urgent_rate"] * 100 for r in rows]
        normal = [r["mean_normal_rate"] * 100 for r in rows]

        axes[0].errorbar(ns, rates, yerr=ci, label=sname,
                         color=colors[sname], marker=markers[sname],
                         capsize=3, linewidth=2, markersize=7)
        axes[1].plot(ns, urgent, label=sname,
                     color=colors[sname], marker=markers[sname],
                     linewidth=2, markersize=7)
        axes[2].plot(ns, normal, label=sname,
                     color=colors[sname], marker=markers[sname],
                     linewidth=2, markersize=7)

    for ax, title in zip(axes, [
        "(a) Overall Service Rate",
        "(b) Urgent Service Rate",
        "(c) Normal Service Rate",
    ]):
        ax.set_xlabel("Number of Requests (N)", fontsize=12)
        ax.set_ylabel("Service Rate (%)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(N_VALUES)
        ax.set_ylim(0, 105)

    fig.suptitle("Algorithm Comparison: Service Rate vs Demand Size",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "fig_algorithm_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
