"""
Unassigned-Node Diagnosis at N=60
=================================
針對 exp_route_comparison.main() 的同一 instance (seed=42, N=60,
fleet 3 SLOW + 2 FAST + 1 UAV)，統計三個演算法的未服務原因：

  TIGHT_TW          : 實體不可能 — 即使單獨派一台空車從 depot 出發，
                      也來不及在 due_date 前完成 (TW + 距離 + 充電時間 卡死)
  SCHEDULE_CONFLICT : 標準上可行，但該演算法已將其他較高優先序請求塞進所有路徑，
                      使此節點再也找不到可行位置 (演算法/排序限制)

第一類為演算法無法改變的「硬限制」(三個演算法理應一致)，
第二類則是演算法策略差異會體現的部分。

Output:
    results/unassigned_diagnosis/diagnosis.csv
    終端統計表
"""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import Route
from exp_route_comparison import run_single_experiment


N_REQUESTS = 60
SEED = 42
OUTPUT_DIR = Path("results") / "unassigned_diagnosis"


# ================================================================
#  分類核心
# ================================================================
def diagnose(problem, node) -> str:
    """傳回 'TIGHT_TW' 或 'SCHEDULE_CONFLICT'。"""

    # 在「全空車隊」上嘗試插入：MCS-SLOW / MCS-FAST 各一次
    for vtype in ('mcs_slow', 'mcs_fast'):
        empty = Route(vehicle_type=vtype, vehicle_id=-1)
        feasible, _ = problem.incremental_insertion_check(empty, 0, node)
        if feasible:
            return "SCHEDULE_CONFLICT"

    # urgent 才有資格試 UAV
    if node.node_type == 'urgent' and problem.compute_uav_delivery(node) > 0:
        empty_uav = Route(vehicle_type='uav', vehicle_id=-1)
        feasible, _ = problem.incremental_insertion_check(empty_uav, 0, node)
        if feasible:
            return "SCHEDULE_CONFLICT"

    return "TIGHT_TW"


def standalone_feasibility(problem, node) -> dict:
    """傳回各車型獨立可行性 (用於細節欄位)。"""
    flags = {}
    for vtype in ('mcs_slow', 'mcs_fast'):
        empty = Route(vehicle_type=vtype, vehicle_id=-1)
        f, _ = problem.incremental_insertion_check(empty, 0, node)
        flags[vtype] = f
    if node.node_type == 'urgent' and problem.compute_uav_delivery(node) > 0:
        empty_uav = Route(vehicle_type='uav', vehicle_id=-1)
        f, _ = problem.incremental_insertion_check(empty_uav, 0, node)
        flags['uav'] = f
    else:
        flags['uav'] = None  # 不適用
    return flags


def depot_distance_manhattan(problem, node) -> float:
    return abs(node.x - problem.depot.x) + abs(node.y - problem.depot.y)


def classify_subreason(flags: dict) -> str:
    """根據各車型獨立可行性細分原因類別。

    A_absolute  : Slow ✗ Fast ✗ UAV ✗   (任何車單獨都不可)
    B_uav_only  : Slow ✗ Fast ✗ UAV ✓   (只 UAV 能救 — TW 對 MCS 太緊)
    C_slow_tight: Slow ✗ Fast ✓         (TW 對 Slow 太緊，Fast/UAV 可)
    D_resource  : Slow ✓                (任何車型都能單獨完成 — 純資源不足)
    """
    s = flags.get('mcs_slow') is True
    f = flags.get('mcs_fast') is True
    u = flags.get('uav') is True
    if s:
        return 'D_resource'
    if f:
        return 'C_slow_tight'
    if u:
        return 'B_uav_only'
    return 'A_absolute'


# ================================================================
#  繪圖
# ================================================================
ALGO_ORDER = ('NN', 'Greedy', 'ALNS')

# 兩大類顏色
COLOR_TW_TIGHT   = '#F4A582'   # 紅 — TW-tight (B + C 合併)
COLOR_D_RESOURCE = '#4393C3'   # 藍 — Resource shortage (D)


def plot_unassigned_reason(per_algo_subcat: dict) -> None:
    """單圖：每個演算法一根堆疊長條，TW-tight (下) + Resource-shortage (上)。"""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    algos = list(ALGO_ORDER)
    x_pos = np.arange(len(algos))
    width = 0.55

    tw_vals = np.array([per_algo_subcat[a]['B_uav_only'] +
                        per_algo_subcat[a]['C_slow_tight'] for a in algos])
    d_vals = np.array([per_algo_subcat[a]['D_resource'] for a in algos])

    ax.bar(x_pos, tw_vals, width,
           color=COLOR_TW_TIGHT, edgecolor='black', linewidth=0.6,
           label='TW-tight')
    ax.bar(x_pos, d_vals, width, bottom=tw_vals,
           color=COLOR_D_RESOURCE, edgecolor='black', linewidth=0.6,
           label='Resource shortage')

    for i, (tw, d) in enumerate(zip(tw_vals, d_vals)):
        if tw > 0:
            ax.text(x_pos[i], tw / 2, f'{tw}',
                    ha='center', va='center', fontsize=11,
                    color='black', fontweight='bold')
        if d > 0:
            ax.text(x_pos[i], tw + d / 2, f'{d}',
                    ha='center', va='center', fontsize=11,
                    color='white', fontweight='bold')

    totals = tw_vals + d_vals
    y_top = max(totals.max(), 1) * 1.15
    for i, t in enumerate(totals):
        ax.text(x_pos[i], t + y_top * 0.02, f'{t}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(algos, fontsize=11)
    ax.set_ylabel('Unassigned Urgent Customers', fontsize=12)
    ax.set_title('Unassigned-Reason Breakdown',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, y_top)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    out_path = OUTPUT_DIR / "fig_unassigned_reason.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ================================================================
#  主程式
# ================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"  Unassigned-Node Diagnosis — N={N_REQUESTS}, seed={SEED}")
    print(f"  Fleet: 3 SLOW + 2 FAST + 1 UAV")
    print("=" * 88)

    solutions, problem = run_single_experiment(N_REQUESTS, seed=SEED)

    rows = []
    summary_rows = []
    per_algo_subcat = {a: Counter() for a in ALGO_ORDER}

    for algo_name in ALGO_ORDER:
        sol = solutions[algo_name]
        served = sum(len(r.nodes) for r in sol.get_all_routes())
        unassigned = sol.unassigned_nodes

        cat_total = Counter()
        cat_urgent = Counter()
        cat_normal = Counter()

        for node in unassigned:
            cat = diagnose(problem, node)
            flags = standalone_feasibility(problem, node)
            sub = classify_subreason(flags)
            cat_total[cat] += 1
            (cat_urgent if node.node_type == 'urgent' else cat_normal)[cat] += 1
            if node.node_type == 'urgent':
                per_algo_subcat[algo_name][sub] += 1

            rows.append({
                'algorithm': algo_name,
                'node_id': node.id,
                'node_type': node.node_type,
                'reason': cat,
                'subreason': sub,
                'demand_kwh': round(node.demand, 2),
                'ready_time': round(node.ready_time, 1),
                'due_date': round(node.due_date, 1),
                'tw_width': round(node.due_date - node.ready_time, 1),
                'depot_dist_km': round(depot_distance_manhattan(problem, node), 2),
                'feasible_alone_slow': flags['mcs_slow'],
                'feasible_alone_fast': flags['mcs_fast'],
                'feasible_alone_uav': flags['uav'],
            })

        # 每個演算法的彙總
        n_unassigned = len(unassigned)
        n_urgent_unassigned = sum(1 for n in unassigned if n.node_type == 'urgent')
        n_normal_unassigned = n_unassigned - n_urgent_unassigned

        print(f"\n[{algo_name}]  served={served}/{N_REQUESTS}  unassigned={n_unassigned}")
        print(f"  By type   : urgent={n_urgent_unassigned}, normal={n_normal_unassigned}")
        print(f"  By reason : TIGHT_TW={cat_total['TIGHT_TW']}, "
              f"SCHEDULE_CONFLICT={cat_total['SCHEDULE_CONFLICT']}")
        print(f"    └ urgent: TIGHT_TW={cat_urgent['TIGHT_TW']}, "
              f"SCHEDULE_CONFLICT={cat_urgent['SCHEDULE_CONFLICT']}")
        print(f"    └ normal: TIGHT_TW={cat_normal['TIGHT_TW']}, "
              f"SCHEDULE_CONFLICT={cat_normal['SCHEDULE_CONFLICT']}")

        summary_rows.append({
            'algorithm': algo_name,
            'served': served,
            'unassigned': n_unassigned,
            'urgent_unassigned': n_urgent_unassigned,
            'normal_unassigned': n_normal_unassigned,
            'tight_tw_total': cat_total['TIGHT_TW'],
            'tight_tw_urgent': cat_urgent['TIGHT_TW'],
            'tight_tw_normal': cat_normal['TIGHT_TW'],
            'schedule_conflict_total': cat_total['SCHEDULE_CONFLICT'],
            'schedule_conflict_urgent': cat_urgent['SCHEDULE_CONFLICT'],
            'schedule_conflict_normal': cat_normal['SCHEDULE_CONFLICT'],
        })

    # CSV 輸出
    detail_path = OUTPUT_DIR / "unassigned_detail.csv"
    if rows:
        with open(detail_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Saved] {detail_path}")

    summary_path = OUTPUT_DIR / "unassigned_summary.csv"
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[Saved] {summary_path}")

    # 列出細分類統計
    print()
    print("=" * 88)
    print("Sub-category breakdown (urgent only)")
    print("=" * 88)
    print(f"{'algo':>8} {'B_uav_only':>12} {'C_slow_tight':>14} {'D_resource':>12} {'A_absolute':>12}")
    for a in ALGO_ORDER:
        c = per_algo_subcat[a]
        print(f"{a:>8} {c['B_uav_only']:>12} {c['C_slow_tight']:>14} "
              f"{c['D_resource']:>12} {c['A_absolute']:>12}")

    plot_unassigned_reason(per_algo_subcat)


if __name__ == '__main__':
    main()
