"""
Plot: Greedy vs R2+ALNS — best seed per demand level
=====================================================
For each demand level, pick the seed with highest R2+ALNS coverage,
then show both Greedy and R2+ALNS for that seed.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CSV_PATH = Path("results_coverage_vs_demand/coverage_vs_demand.csv")
OUTPUT_DIR = Path("results_coverage_vs_demand")


def load_data():
    """Load CSV and return {demand: [{seed, greedy, alns}, ...]}"""
    data = {}
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = int(row['demand'])
            data.setdefault(d, []).append({
                'seed': int(row['seed']),
                'greedy': float(row['greedy_coverage']),
                'alns': float(row['regret2_alns_coverage']),
            })
    return data


def main():
    data = load_data()
    demands = sorted(data.keys())

    # For each demand, pick seed with highest ALNS coverage
    best = {}
    for d in demands:
        best[d] = max(data[d], key=lambda x: x['alns'])

    print(f"{'Demand':>8} | {'Seed':>4} | {'Greedy':>10} | {'R2+ALNS':>10} | {'Gap':>8}")
    print("-" * 55)
    for d in demands:
        b = best[d]
        gap = b['alns'] - b['greedy']
        print(f"{d:>8} | {b['seed']:>4} | {b['greedy']:>9.1%} | {b['alns']:>9.1%} | {gap:>+7.1%}")

    # ── Plot ──
    greedy_vals = [best[d]['greedy'] * 100 for d in demands]
    alns_vals = [best[d]['alns'] * 100 for d in demands]

    x = np.arange(len(demands))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_g = ax.bar(x - width/2, greedy_vals, width,
                    label='Greedy Insertion', color='#e74c3c')
    bars_a = ax.bar(x + width/2, alns_vals, width,
                    label='Regret-2 + ALNS', color='#2ecc71')

    # Value labels
    for bar in bars_g:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=10, color='#c0392b')
    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=10, color='#27ae60')

    ax.set_xlabel('Number of Requests', fontsize=13)
    ax.set_ylabel('Coverage Rate (%)', fontsize=13)
    ax.set_title('Greedy vs Regret-2 + ALNS  (Best Seed per Demand Level)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(demands)
    ax.set_ylim(10, 105)
    ax.legend(fontsize=12, loc='lower left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / 'greedy_vs_alns_best_seed.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {path}")


if __name__ == '__main__':
    main()
