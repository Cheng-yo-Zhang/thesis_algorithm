"""
Coverage Rate vs Demand (100 seeds, parallel) — fig_coverage_vs_demand.png
==========================================================================
Multi-process 平行版：每個 (seed, N) 由獨立 worker 跑，append 寫進 progress.csv，
全部完成後再彙整成 service_rate_raw.csv (wide) + summary + plot。

繼承自 exp_route_comparison.py 的圖 A：
  Fleet      : 3 SLOW + 2 FAST + 1 UAV
  Algorithms : NN | Greedy | ALNS (Greedy initial, 5000 iter)
  N          : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

特性：
  - 並行：ProcessPoolExecutor (default 10 workers，可用 --workers 調)
  - LPT 排程：先丟最耗時的 N=100 task，減少 tail effect
  - Crash-safe：每完成一個 (seed, N) 就 append 到 progress.csv
  - Resume：重跑時自動讀 progress.csv 跳過已完成的 (seed, N)

用法：
  python exp_coverage_100seeds.py --seeds 1                  # 快速測試
  python exp_coverage_100seeds.py --seeds 100                # 正式跑 (10 workers)
  python exp_coverage_100seeds.py --seeds 100 --workers 12   # 全核滿載
"""

import argparse
import csv
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alns import ALNSSolver
from config import Config
from problem import ChargingSchedulingProblem
from simulation import initialize_fleet


# ================================================================
#  實驗參數
# ================================================================
DEMAND_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
BASE_SEED = 1
NUM_MCS_SLOW = 3
NUM_MCS_FAST = 2
NUM_UAV = 1
ALNS_ITER = 5000
OUTPUT_DIR = Path("results") / "coverage_100seeds"

ALGO_NAMES = ['NN', 'TCGI', 'TCGI-ALNS']
ALGO_COLORS = {'NN': '#FF9800', 'TCGI': '#E53935', 'TCGI-ALNS': '#1E88E5'}
ALGO_MARKERS = {'NN': '^', 'TCGI': 'o', 'TCGI-ALNS': 's'}


# ================================================================
#  單次求解 (對單一 seed × N 跑三種演算法)
# ================================================================
def solve_one_instance(n_demand: int, seed: int) -> dict:
    cfg = Config(
        RANDOM_SEED=seed,
        N_REQUESTS=n_demand,
        CONSTRUCTION_STRATEGY="regret2",  # 僅作為 greedy 的 fallback 排序鍵
        NUM_MCS_SLOW=NUM_MCS_SLOW,
        NUM_MCS_FAST=NUM_MCS_FAST,
        NUM_UAV=NUM_UAV,
        MCS_UNLIMITED_ENERGY=True,
        ALNS_MAX_ITERATIONS=ALNS_ITER,
    )

    problem = ChargingSchedulingProblem(cfg)
    fleet = initialize_fleet(cfg, problem.depot)
    requests = problem.generate_requests()
    problem.setup_nodes(requests)

    def reset():
        random.seed(seed)
        np.random.seed(seed)
        for n in requests:
            n.status = 'new'

    reset()
    nn_sol = problem.nearest_neighbor_construction(list(requests), fleet)

    reset()
    greedy_sol = problem.greedy_insertion_construction(list(requests), fleet)

    reset()
    greedy_for_alns = problem.greedy_insertion_construction(list(requests), fleet)
    alns_sol = ALNSSolver(problem, cfg).solve(greedy_for_alns)

    return {'NN': nn_sol, 'TCGI': greedy_sol, 'TCGI-ALNS': alns_sol}


def extract_metrics(sol, n_demand: int) -> dict:
    n_served = sum(len(r.nodes) for r in sol.get_all_routes())
    n_missed = len(sol.unassigned_nodes)
    n_total = n_served + n_missed
    return {
        'n_total': n_total,
        'n_served': n_served,
        'n_missed': n_missed,
        'service_rate': n_served / n_total if n_total > 0 else 1.0,
    }


# ================================================================
#  Worker entry + progress.csv (long format) 的 I/O
# ================================================================
PROGRESS_FIELDS = ['seed', 'n_demand', 'algorithm', 'n_total', 'n_served',
                   'n_missed', 'service_rate', 'elapsed_sec']


def _run_task(args):
    """Worker entry point — 必須是 module-level 才能 pickle。

    Args:
        args: (seed, n_demand) tuple
    Returns:
        list[dict] — 每個演算法一個 dict (共 3 個)
    """
    seed, n_demand = args
    t0 = time.time()
    sols = solve_one_instance(n_demand, seed)
    elapsed = time.time() - t0

    out = []
    for algo in ALGO_NAMES:
        m = extract_metrics(sols[algo], n_demand)
        out.append({
            'seed': seed,
            'n_demand': n_demand,
            'algorithm': algo,
            'n_total': m['n_total'],
            'n_served': m['n_served'],
            'n_missed': m['n_missed'],
            'service_rate': m['service_rate'],
            'elapsed_sec': elapsed,
        })
    return out


def _load_completed_pairs(progress_path: Path) -> set:
    """讀 progress.csv，回傳「三個演算法都跑完」的 (seed, n_demand) 集合。"""
    if not progress_path.exists():
        return set()
    by_pair: dict = {}
    with open(progress_path, 'r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (int(row['seed']), int(row['n_demand']))
            by_pair.setdefault(key, set()).add(row['algorithm'])
    target = set(ALGO_NAMES)
    return {k for k, algos in by_pair.items() if algos >= target}


def _append_progress(progress_path: Path, rows: list) -> None:
    """Append 一個 task 的 3 個演算法結果到 progress.csv。"""
    is_new = not progress_path.exists()
    with open(progress_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_raw_rows_from_progress(progress_path: Path) -> list:
    """讀 progress.csv，dedupe by (seed, n_demand, algorithm) 取最新一筆，
    回傳 save_raw_csv / save_summary_csv / plot_service_rate 認識的格式。"""
    seen: dict = {}
    with open(progress_path, 'r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (int(row['seed']), int(row['n_demand']), row['algorithm'])
            seen[key] = {
                'seed': int(row['seed']),
                'n_demand': int(row['n_demand']),
                'algorithm': row['algorithm'],
                'n_total': int(row['n_total']),
                'n_served': int(row['n_served']),
                'n_missed': int(row['n_missed']),
                'service_rate': float(row['service_rate']),
            }
    return list(seen.values())


# ================================================================
#  CSV 輸出
# ================================================================
def save_raw_csv(rows: list, path: Path) -> None:
    """寬格式 CSV — 每個 seed 一列；欄位群組 = (N 值 × 3 種演算法) + 空白欄分隔。

    版面：
        Row 1 : (空白)
        Row 2 : ,10,,,,20,,,,30,,,, ... ,80,,
        Row 3 : ,ALNS,Greedy,NN,,ALNS,Greedy,NN,, ...
        Row 4+: <seed>,<alns>,<greedy>,<nn>,,<alns>,<greedy>,<nn>,, ...
    """
    # 收進 (seed, n_demand, algo) → service_rate 的快取
    lookup: dict = {}
    seeds: set = set()
    for row in rows:
        lookup[(row['seed'], row['n_demand'], row['algorithm'])] = row['service_rate']
        seeds.add(row['seed'])
    sorted_seeds = sorted(seeds)

    # 標頭：每個 N 群組 = 3 個演算法欄 + 1 個空白分隔欄（最末群組仍保留空白欄以利對齊）
    header_n: list = ['']
    header_algo: list = ['']
    for n in DEMAND_LEVELS:
        header_n += [str(n), '', '', '']
        header_algo += ['TCGI-ALNS', 'TCGI', 'NN', '']

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])             # Row 1: 空白
        writer.writerow(header_n)       # Row 2: N 值（僅放在群組首欄）
        writer.writerow(header_algo)    # Row 3: 演算法名稱

        for seed in sorted_seeds:
            row_out: list = [seed]
            for n in DEMAND_LEVELS:
                row_out.append(lookup.get((seed, n, 'TCGI-ALNS'), ''))
                row_out.append(lookup.get((seed, n, 'TCGI'), ''))
                row_out.append(lookup.get((seed, n, 'NN'), ''))
                row_out.append('')      # 群組間空白欄
            writer.writerow(row_out)


def save_summary_csv(raw_rows: list, path: Path) -> None:
    """彙總 raw rows → (n_demand, algorithm) 的 mean / std / min / max。"""
    by_key: dict = {}
    for row in raw_rows:
        key = (row['n_demand'], row['algorithm'])
        by_key.setdefault(key, []).append(row['service_rate'])

    fieldnames = ['n_demand', 'algorithm', 'n_seeds',
                  'mean_service_rate', 'std_service_rate',
                  'min_service_rate', 'max_service_rate']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for n in DEMAND_LEVELS:
            for algo in ALGO_NAMES:
                rates = by_key.get((n, algo), [])
                if not rates:
                    continue
                writer.writerow({
                    'n_demand': n,
                    'algorithm': algo,
                    'n_seeds': len(rates),
                    'mean_service_rate': float(np.mean(rates)),
                    'std_service_rate': float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
                    'min_service_rate': float(np.min(rates)),
                    'max_service_rate': float(np.max(rates)),
                })


# ================================================================
#  繪圖（與 exp_route_comparison.py 圖 A 一致：無 error bar）
# ================================================================
def plot_service_rate(raw_rows: list, path: Path) -> None:
    by_key: dict = {}
    for row in raw_rows:
        key = (row['n_demand'], row['algorithm'])
        by_key.setdefault(key, []).append(row['service_rate'])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in ALGO_NAMES:
        means = [np.mean(by_key.get((n, name), [0.0])) * 100 for n in DEMAND_LEVELS]
        ax.plot(DEMAND_LEVELS, means,
                marker=ALGO_MARKERS[name], color=ALGO_COLORS[name],
                linewidth=2, markersize=8, label=name, zorder=5)

    ax.set_xlabel('Number of Requests (N)', fontsize=12)
    ax.set_ylabel('Service Rate (%)', fontsize=12)
    ax.set_xticks(DEMAND_LEVELS)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================================================================
#  主程式
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Coverage Rate vs Demand (multi-seed, parallel)")
    parser.add_argument('--seeds', type=int, default=100,
                        help='Number of seeds to run (default: 100).')
    parser.add_argument('--workers', type=int, default=10,
                        help='Parallel worker processes (default: 10). '
                             'i7-12700 has 12 physical cores; use 10 to keep '
                             '2 cores free for OS / IDE.')
    args = parser.parse_args()
    num_seeds = args.seeds
    n_workers = max(1, args.workers)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = OUTPUT_DIR / 'progress.csv'

    # ── Task list with LPT scheduling (大 N 先丟，減少 tail) ──
    all_tasks = [(BASE_SEED + rep, n)
                 for rep in range(num_seeds)
                 for n in DEMAND_LEVELS]
    all_tasks.sort(key=lambda t: (-t[1], t[0]))  # N 降序，seed 升序

    completed = _load_completed_pairs(progress_path)
    pending = [t for t in all_tasks if t not in completed]
    total = len(all_tasks)
    n_done = len(completed)

    print("=" * 72)
    print("  Coverage Rate vs Demand (multi-seed, parallel)")
    print(f"  Fleet     : {NUM_MCS_SLOW} SLOW + {NUM_MCS_FAST} FAST + {NUM_UAV} UAV")
    print(f"  Algorithms: {' | '.join(ALGO_NAMES)} (ALNS = Greedy init + {ALNS_ITER} iter)")
    print(f"  N         : {DEMAND_LEVELS}")
    print(f"  Seeds     : {num_seeds} (base seed = {BASE_SEED})")
    print(f"  Workers   : {n_workers}")
    print(f"  Progress  : {progress_path}")
    print(f"  Total runs: {total}  (already done: {n_done}, pending: {len(pending)})")
    print("=" * 72)

    t_start = time.time()
    failures: list = []

    if pending:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            fut_to_task = {executor.submit(_run_task, t): t for t in pending}

            for fut in as_completed(fut_to_task):
                task = fut_to_task[fut]
                try:
                    rows = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  [FAIL] seed={task[0]:>3} N={task[1]:>3}  "
                          f"{type(e).__name__}: {e}")
                    failures.append((task, repr(e)))
                    continue

                _append_progress(progress_path, rows)
                n_done += 1

                seed, n = task
                rates = {r['algorithm']: r['service_rate'] for r in rows}
                inst_sec = rows[0]['elapsed_sec']
                done_this_run = n_done - len(completed)
                cum_min = (time.time() - t_start) / 60
                eta_min = (cum_min / done_this_run * (total - n_done)
                           if done_this_run > 0 else 0.0)
                print(f"  [{n_done:>4}/{total}] seed={seed:>3} N={n:>3}  "
                      f"NN={rates['NN']:5.1%}  TCGI={rates['TCGI']:5.1%}  "
                      f"TCGI-ALNS={rates['TCGI-ALNS']:5.1%}  ")
    else:
        print("  (Nothing to run — all tasks already completed in progress.csv.)")

    # ── 從 progress.csv 重組成 wide format + summary + 圖 ──
    raw_rows = _load_raw_rows_from_progress(progress_path)
    raw_path = OUTPUT_DIR / 'service_rate_raw.csv'
    summary_path = OUTPUT_DIR / 'service_rate_summary.csv'
    fig_path = OUTPUT_DIR / 'fig_coverage_vs_demand.png'

    save_raw_csv(raw_rows, raw_path)
    save_summary_csv(raw_rows, summary_path)
    plot_service_rate(raw_rows, fig_path)

    total_sec = time.time() - t_start
    print("=" * 72)
    print(f"  Wall time   : {total_sec/60:.1f} min  ({total_sec/3600:.2f} h)")
    print(f"  Completed   : {n_done}/{total}")
    if failures:
        print(f"  FAILURES    : {len(failures)}  (see lines marked [FAIL] above)")
        for task, err in failures[:5]:
            print(f"    seed={task[0]} N={task[1]}: {err}")
    print(f"  Progress CSV: {progress_path}")
    print(f"  Raw CSV     : {raw_path}")
    print(f"  Summary CSV : {summary_path}")
    print(f"  Figure      : {fig_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
