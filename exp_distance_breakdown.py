"""
Distance Breakdown by Vehicle Type — fixed N single-shot comparison.

Fixed:
    N            = 50 customers (static batch)
    fleet        = 3 Slow + 2 Fast + 1 UAV
    seed         = 42 (single replication)
    ALNS iter    = 5000 (ALNS uses Greedy-EDF as the initial solution)

Compare NN / Greedy / ALNS on:
    total travel distance split by vehicle type (Slow / Fast / UAV)

Output:
    results_distance_breakdown/raw.csv
    results_distance_breakdown/fig_distance_breakdown_by_type.png
"""

import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp_route_comparison import run_single_experiment, get_metrics


# === Experiment Parameters ===
N_CUSTOMERS = 50
SEED = 42
ALGO_NAMES = ['NN', 'Greedy', 'ALNS']

TYPE_COLORS = {
    'slow': '#1E88E5',
    'fast': '#1E88E5',
    'uav':  '#1E88E5',
}

OUTPUT_DIR = Path("results_distance_breakdown")


# ================================================================
#  Run
# ================================================================
def run() -> list:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Distance Breakdown — N={N_CUSTOMERS}, seed={SEED}")
    print("=" * 72)

    t0 = time.time()
    solutions, _problem = run_single_experiment(N_CUSTOMERS, seed=SEED)
    elapsed = time.time() - t0
    print(f"run_single_experiment: {elapsed:.1f}s\n")

    rows = []
    for name in ALGO_NAMES:
        m = get_metrics(solutions[name])
        row = {
            "algorithm":      name,
            "n_customers":    N_CUSTOMERS,
            "seed":           SEED,
            "coverage":       m['coverage'],
            "served":         m['served'],
            "missed":         m['missed'],
            "slow_distance":  m['slow_distance'],
            "fast_distance":  m['fast_distance'],
            "uav_distance":   m['uav_distance'],
            "mcs_distance":   m['mcs_distance'],
            "total_distance": m['distance'],
        }
        rows.append(row)
        print(
            f"  {name:<8s} cov={m['coverage']:5.1f}%  "
            f"slow={m['slow_distance']:6.1f} km  "
            f"fast={m['fast_distance']:6.1f} km  "
            f"uav={m['uav_distance']:5.1f} km  "
            f"total={m['distance']:6.1f} km"
        )

    csv_path = OUTPUT_DIR / "raw.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {csv_path}")

    return rows


# ================================================================
#  Plot
# ================================================================
def plot(rows: list) -> None:
    slow_vals = np.array([r['slow_distance'] for r in rows])
    fast_vals = np.array([r['fast_distance'] for r in rows])
    uav_vals  = np.array([r['uav_distance']  for r in rows])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    x_pos = np.arange(len(ALGO_NAMES))
    width = 0.55

    ax.bar(x_pos, slow_vals, width, color=TYPE_COLORS['slow'])
    ax.bar(x_pos, fast_vals, width, bottom=slow_vals, color=TYPE_COLORS['fast'])
    ax.bar(x_pos, uav_vals, width, bottom=slow_vals + fast_vals, color=TYPE_COLORS['uav'])

    totals = slow_vals + fast_vals + uav_vals
    y_top = totals.max() if totals.max() > 0 else 1.0
    for i, total in enumerate(totals):
        ax.text(x_pos[i], total + y_top * 0.015,
                f'{total:.0f} km', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(ALGO_NAMES, fontsize=11)
    ax.set_ylabel('Total Distance (km)', fontsize=12)
    ax.set_title(f'Distance\n'
                 '[Fleet: 3 SLOW + 2 FAST + 1 UAV]',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(top=y_top * 1.12)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    png_path = OUTPUT_DIR / "fig_distance_breakdown_by_type.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {png_path}")


if __name__ == "__main__":
    rows = run()
    plot(rows)
