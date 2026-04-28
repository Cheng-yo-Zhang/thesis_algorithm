"""
Figure 6: Algorithm Comparison — Service Rate vs Lambda
========================================================
x-axis: λ (demand intensity, requests/min)
y-axis: MW Service Rate (%)
Lines:  Greedy (EDF) | Regret-2 | Regret-2 + ALNS

Mode: Poisson arrival (USE_FIXED_DEMAND=False, USE_HPP=True)
Fleet: 20 Slow + 20 Fast + 20 UAV (fixed)
"""

import numpy as np
import csv
import time
import json
from math import sqrt
from pathlib import Path

from config import Config
from main import run_simulation


# === Experiment Parameters ===
LAMBDA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
NUM_REPLICATIONS = 1
BASE_SEED = 42
OUTPUT_DIR = Path("results_algorithm_comparison")

STRATEGIES = [
    {"name": "Greedy (EDF)",    "construction": "edf",     "alns_iter": 0},
    {"name": "Regret-2",        "construction": "regret2", "alns_iter": 0},
    {"name": "Regret-2 + ALNS", "construction": "regret2", "alns_iter": 5000},
]


def make_cfg(lam: float, strategy: dict, seed: int) -> Config:
    return Config(
        RANDOM_SEED=seed,
        USE_FIXED_DEMAND=False,
        USE_HPP=True,
        LAMBDA_BAR=lam,
        CONSTRUCTION_STRATEGY=strategy["construction"],
        ALNS_MAX_ITERATIONS=strategy["alns_iter"],
    )


def ci95(arr):
    m = np.mean(arr)
    s = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    h = 1.96 * s / sqrt(len(arr)) if len(arr) > 1 else 0.0
    return m, s, h


def run_experiment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    results = []

    total_runs = len(LAMBDA_VALUES) * len(STRATEGIES) * NUM_REPLICATIONS
    run_count = 0

    print(f"{'='*80}")
    print(f"Figure 6: Algorithm Comparison — Service Rate vs Lambda")
    print(f"{'='*80}")
    print(f"  Lambda values: {LAMBDA_VALUES}")
    print(f"  Strategies: {[s['name'] for s in STRATEGIES]}")
    print(f"  Replications: {NUM_REPLICATIONS}")
    print(f"  Total runs: {total_runs}")
    print()

    for lam in LAMBDA_VALUES:
        for strategy in STRATEGIES:
            rep_results = {
                "service_rate": [],
                "urgent_rate": [],
                "normal_rate": [],
                "mw_total": [],
                "mw_served": [],
                "fleet_used": [],
                "n_slow": [],
                "n_fast": [],
                "n_uav": [],
            }

            for rep in range(NUM_REPLICATIONS):
                cfg = make_cfg(lam, strategy, BASE_SEED + rep)
                _, _, _, _, stats = run_simulation(cfg, verbose=False)

                rep_results["service_rate"].append(stats["mw_service_rate"])
                rep_results["urgent_rate"].append(stats["mw_urgent_rate"])
                rep_results["normal_rate"].append(stats["mw_normal_rate"])
                rep_results["mw_total"].append(stats["mw_total"])
                rep_results["mw_served"].append(stats["mw_served"])
                rep_results["fleet_used"].append(stats["mw_fleet_used"])
                rep_results["n_slow"].append(stats["mw_n_slow"])
                rep_results["n_fast"].append(stats["mw_n_fast"])
                rep_results["n_uav"].append(stats["mw_n_uav"])

                run_count += 1

            sr_m, sr_s, sr_h = ci95(rep_results["service_rate"])
            ur_m, _, _ = ci95(rep_results["urgent_rate"])
            nr_m, _, _ = ci95(rep_results["normal_rate"])
            fl_m, _, _ = ci95(rep_results["fleet_used"])

            row = {
                "lambda": lam,
                "strategy": strategy["name"],
                "mean_service_rate": sr_m,
                "std_service_rate": sr_s,
                "ci95_half": sr_h,
                "mean_urgent_rate": ur_m,
                "mean_normal_rate": nr_m,
                "mean_mw_total": np.mean(rep_results["mw_total"]),
                "mean_mw_served": np.mean(rep_results["mw_served"]),
                "mean_fleet_used": fl_m,
                "mean_n_slow": np.mean(rep_results["n_slow"]),
                "mean_n_fast": np.mean(rep_results["n_fast"]),
                "mean_n_uav": np.mean(rep_results["n_uav"]),
            }
            results.append(row)

            elapsed = time.time() - t_start
            print(f"  [{run_count:3d}/{total_runs}] λ={lam:.2f} {strategy['name']:>17s} | "
                  f"rate={sr_m:.3f}±{sr_s:.3f} | "
                  f"urgent={ur_m:.3f} normal={nr_m:.3f} | "
                  f"fleet={fl_m:.1f} [{elapsed:.0f}s]")

    # Save CSV
    csv_path = OUTPUT_DIR / "algorithm_comparison.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Save JSON (for plotting)
    json_path = OUTPUT_DIR / "algorithm_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    total_time = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    print(f"  {'Lambda':>6} {'Strategy':>17} {'Rate':>8} {'±CI95':>7} "
          f"{'Urgent':>8} {'Normal':>8} {'Fleet':>6}")
    print(f"  {'-'*6} {'-'*17} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['lambda']:6.2f} {r['strategy']:>17s} {r['mean_service_rate']:8.3f} "
              f"±{r['ci95_half']:5.3f} {r['mean_urgent_rate']:8.3f} "
              f"{r['mean_normal_rate']:8.3f} {r['mean_fleet_used']:6.1f}")
    print(f"\nTotal time: {total_time/60:.1f} min")
    print(f"Results saved to {csv_path}")

    # Plot
    plot_results(results)
    return results


def plot_results(results):
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
        lams = [r["lambda"] for r in rows]
        rates = [r["mean_service_rate"] * 100 for r in rows]
        ci = [r["ci95_half"] * 100 for r in rows]
        urgent = [r["mean_urgent_rate"] * 100 for r in rows]
        normal = [r["mean_normal_rate"] * 100 for r in rows]

        # (a) Overall Service Rate
        axes[0].errorbar(lams, rates, yerr=ci, label=sname,
                         color=colors[sname], marker=markers[sname],
                         capsize=3, linewidth=2, markersize=7)

        # (b) Urgent Service Rate
        axes[1].plot(lams, urgent, label=sname,
                     color=colors[sname], marker=markers[sname],
                     linewidth=2, markersize=7)

        # (c) Normal Service Rate
        axes[2].plot(lams, normal, label=sname,
                     color=colors[sname], marker=markers[sname],
                     linewidth=2, markersize=7)

    for ax, title in zip(axes, [
        "(a) Overall Service Rate",
        "(b) Urgent Service Rate",
        "(c) Normal Service Rate",
    ]):
        ax.set_xlabel("λ (requests/min)", fontsize=12)
        ax.set_ylabel("Service Rate (%)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(LAMBDA_VALUES[0] - 0.01, LAMBDA_VALUES[-1] + 0.01)
        ax.set_ylim(0, 105)

    fig.suptitle("Algorithm Comparison: Service Rate vs Demand Intensity",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "fig6_algorithm_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
