"""
Urgent Miss Rate vs Urgent Ratio (multi-seed)
=============================================

繼承自 exp_uav_vs_urgent_ratio.py：
  Fixed       : n_requests=20, urgent_tw=30min, fleet=(SLOW=3, FAST=2)
  Construction: regret2 + ALNS (5000 iter)
  Rho grid    : [0.1, 0.2, ..., 0.9]
  NUM_UAV     : [0, 1, 2, 3]                  → 四條線
  Output      : results/uav_vs_urgent_ratio_100seeds/urgent_miss_rate.png

用法：
  python exp_uav_vs_urgent_ratio_100seeds.py --seeds 1    # 測試 CSV 格式
  python exp_uav_vs_urgent_ratio_100seeds.py --seeds 100  # 正式跑（預設）
"""

import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from exp_fleet_vs_demand import generate_static_requests, solve_once
from problem import ChargingSchedulingProblem
from simulation import initialize_fleet


# ================================================================
#  實驗參數（與 exp_uav_vs_urgent_ratio.py 一致）
# ================================================================
URGENT_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_UAV_LIST = [0, 1, 2, 3]
BASE_SEED = 1
N_REQUESTS = 20
NUM_MCS_SLOW = 3
NUM_MCS_FAST = 2
URGENT_TW = 30.0
ALNS_ITER = 5000

OUTPUT_DIR = Path("results") / "uav_vs_urgent_ratio_100seeds"
STRATEGY = {"construction": "regret2", "alns_iter": ALNS_ITER}


# ================================================================
#  單次求解
# ================================================================
def run_one(rho: float, num_uav: int, seed: int) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    cfg = Config(
        RANDOM_SEED=seed,
        URGENT_RATIO=rho,
        URGENT_TW_MIN=URGENT_TW,
        URGENT_TW_MAX=URGENT_TW,
        NUM_MCS_SLOW=NUM_MCS_SLOW,
        NUM_MCS_FAST=NUM_MCS_FAST,
        NUM_UAV=num_uav,
        CONSTRUCTION_STRATEGY="regret2",
        ALNS_MAX_ITERATIONS=ALNS_ITER,
    )
    problem = ChargingSchedulingProblem(cfg)
    fleet = initialize_fleet(cfg, problem.depot)
    requests = generate_static_requests(problem, N_REQUESTS, cfg)
    problem.setup_nodes(requests)

    t0 = time.time()
    solution = solve_once(requests, fleet, problem, cfg, STRATEGY)
    elapsed = time.time() - t0

    solution.calculate_total_cost(total_customers=N_REQUESTS)

    urgent_total = sum(1 for r in requests if r.node_type == 'urgent')
    urgent_missed = sum(1 for n in solution.unassigned_nodes if n.node_type == 'urgent')
    urgent_miss_rate = urgent_missed / urgent_total if urgent_total > 0 else 0.0

    return {
        "seed": seed,
        "rho": rho,
        "num_uav": num_uav,
        "urgent_total": urgent_total,
        "urgent_missed": urgent_missed,
        "urgent_miss_rate": urgent_miss_rate,
        "elapsed_sec": elapsed,
    }


# ================================================================
#  CSV 輸出（寬格式）
# ================================================================
def save_raw_csv(rows: list, path: Path) -> None:
    """寬格式 CSV — 每個 seed 一列；欄位群組 = (rho × 4 UAV) + 空白欄分隔。

    版面：
        Row 1 : (空白)
        Row 2 : ,0.1,,,,,0.2,,,,,...,0.9,,,,
        Row 3 : ,UAV=0,UAV=1,UAV=2,UAV=3,,UAV=0,UAV=1,UAV=2,UAV=3,, ...
        Row 4+: <seed>,<r00>,<r01>,<r02>,<r03>,,<r10>,<r11>,<r12>,<r13>,, ...
    """
    lookup: dict = {}
    seeds: set = set()
    for row in rows:
        lookup[(row['seed'], row['rho'], row['num_uav'])] = row['urgent_miss_rate']
        seeds.add(row['seed'])
    sorted_seeds = sorted(seeds)

    header_rho: list = ['']
    header_uav: list = ['']
    for rho in URGENT_RATIOS:
        header_rho += [f'{rho:.1f}'] + [''] * len(NUM_UAV_LIST)
        header_uav += [f'UAV={k}' for k in NUM_UAV_LIST] + ['']

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])             # Row 1
        writer.writerow(header_rho)     # Row 2
        writer.writerow(header_uav)     # Row 3
        for seed in sorted_seeds:
            row_out: list = [seed]
            for rho in URGENT_RATIOS:
                for k in NUM_UAV_LIST:
                    row_out.append(lookup.get((seed, rho, k), ''))
                row_out.append('')      # 群組間空白欄
            writer.writerow(row_out)


def save_summary_csv(rows: list, path: Path) -> None:
    by_key: dict = {}
    for row in rows:
        by_key.setdefault((row['rho'], row['num_uav']), []).append(row['urgent_miss_rate'])

    fieldnames = ['rho', 'num_uav', 'n_seeds',
                  'mean_miss_rate', 'std_miss_rate',
                  'min_miss_rate', 'max_miss_rate']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rho in URGENT_RATIOS:
            for k in NUM_UAV_LIST:
                rates = by_key.get((rho, k), [])
                if not rates:
                    continue
                writer.writerow({
                    'rho': rho,
                    'num_uav': k,
                    'n_seeds': len(rates),
                    'mean_miss_rate': float(np.mean(rates)),
                    'std_miss_rate': float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
                    'min_miss_rate': float(np.min(rates)),
                    'max_miss_rate': float(np.max(rates)),
                })


# ================================================================
#  繪圖（與 exp_uav_vs_urgent_ratio.py 一致，y 值 = seed 平均）
# ================================================================
def plot_miss_rate(rows: list, path: Path) -> None:
    by_key: dict = {}
    for row in rows:
        by_key.setdefault((row['rho'], row['num_uav']), []).append(row['urgent_miss_rate'])

    styles = {
        0: dict(label="Baseline (No UAV)", linestyle="--", marker="o", color="#7f8c8d"),
        1: dict(label="With UAV (K=1)",    linestyle="-",  marker="s", color="#f39c12"),
        2: dict(label="With UAV (K=2)",    linestyle="-",  marker="^", color="#e74c3c"),
        3: dict(label="With UAV (K=3)",    linestyle="-",  marker="D", color="#8e44ad"),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    for k in NUM_UAV_LIST:
        xs = URGENT_RATIOS
        ys = [float(np.mean(by_key.get((rho, k), [0.0]))) for rho in URGENT_RATIOS]
        ax.plot(xs, ys, linewidth=1.8, markersize=6, **styles[k])

    ax.set_xlabel("Urgent Ratio", fontsize=12)
    ax.set_ylabel("Urgent Miss Rate", fontsize=12)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(bottom=0)
    ax.set_xticks(URGENT_RATIOS)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, linestyle=":", alpha=1.0)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("UAV Impact on Urgent Miss Rate", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ================================================================
#  主程式
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Urgent Miss Rate vs Urgent Ratio (multi-seed)")
    parser.add_argument('--seeds', type=int, default=100,
                        help='Number of seeds to run (default: 100). '
                             'Use --seeds 1 to verify CSV format quickly.')
    args = parser.parse_args()
    num_seeds = args.seeds

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = num_seeds * len(URGENT_RATIOS) * len(NUM_UAV_LIST)
    print("=" * 72)
    print("  Urgent Miss Rate vs Urgent Ratio (multi-seed)")
    print(f"  Fleet      : {NUM_MCS_SLOW} SLOW + {NUM_MCS_FAST} FAST + UAV ∈ {NUM_UAV_LIST}")
    print(f"  Rho grid   : {URGENT_RATIOS}")
    print(f"  N_REQUESTS : {N_REQUESTS}  (urgent_tw={URGENT_TW}min)")
    print(f"  ALNS iter  : {ALNS_ITER}")
    print(f"  Seeds      : {num_seeds} (base seed = {BASE_SEED})")
    print(f"  Total runs : {total_runs}")
    print("=" * 72)

    raw_rows: list = []
    t_start = time.time()
    run_count = 0

    for rep in range(num_seeds):
        seed = BASE_SEED + rep
        for num_uav in NUM_UAV_LIST:
            for rho in URGENT_RATIOS:
                row = run_one(rho, num_uav, seed)
                raw_rows.append(row)
                run_count += 1

                cum = time.time() - t_start
                eta = cum / run_count * (total_runs - run_count)
                print(
                    f"  [{run_count:>5}/{total_runs}] seed={seed:>3} "
                    f"rho={rho:.1f} uav={num_uav} | "
                    f"urgent={row['urgent_total']:>2} miss={row['urgent_missed']:>2} "
                    f"rate={row['urgent_miss_rate']:.3f} | "
                    f"{row['elapsed_sec']:.1f}s (ETA {eta/60:.1f}min)"
                )

    raw_path = OUTPUT_DIR / 'urgent_miss_rate_raw.csv'
    summary_path = OUTPUT_DIR / 'urgent_miss_rate_summary.csv'
    fig_path = OUTPUT_DIR / 'urgent_miss_rate.png'

    save_raw_csv(raw_rows, raw_path)
    save_summary_csv(raw_rows, summary_path)
    plot_miss_rate(raw_rows, fig_path)

    total = time.time() - t_start
    print("=" * 72)
    print(f"  Total time : {total/60:.1f} min")
    print(f"  Raw CSV    : {raw_path}")
    print(f"  Summary CSV: {summary_path}")
    print(f"  Figure     : {fig_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
