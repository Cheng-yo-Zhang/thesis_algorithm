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

# B/C 兩個 TW-tight 的子類別共用紅色系（深淺區分嚴格度）
COLOR_C_SLOW_TIGHT = '#F4A582'   # 淺紅 — TW 對 Slow 太緊 (Fast/UAV 仍可)
COLOR_B_UAV_ONLY   = '#B2182B'   # 深紅 — TW 緊到只 UAV 可救
COLOR_D_RESOURCE   = '#4393C3'   # 藍   — 純資源不足


def plot_unassigned_reason(per_algo_subcat: dict) -> None:
    """並排雙圖：左 = TW-tight (B+C 堆疊)，右 = Resource-shortage (D)。"""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.5),
                                            gridspec_kw={'wspace': 0.25})

    algos = list(ALGO_ORDER)
    x_pos = np.arange(len(algos))
    width = 0.55

    # === 左圖: TW-tight (C 在下、B 在上) =========================
    c_vals = np.array([per_algo_subcat[a]['C_slow_tight'] for a in algos])
    b_vals = np.array([per_algo_subcat[a]['B_uav_only']   for a in algos])

    bar_c = ax_left.bar(x_pos, c_vals, width,
                        color=COLOR_C_SLOW_TIGHT, edgecolor='black',
                        linewidth=0.6,
                        label='C: TW Slow-tight (Fast/UAV feasible)')
    bar_b = ax_left.bar(x_pos, b_vals, width, bottom=c_vals,
                        color=COLOR_B_UAV_ONLY, edgecolor='black',
                        linewidth=0.6,
                        label='B: TW MCS-tight (UAV-only feasible)')

    # 每段中央標數字
    for i, (c, b) in enumerate(zip(c_vals, b_vals)):
        if c > 0:
            ax_left.text(x_pos[i], c / 2, f'{c}',
                         ha='center', va='center', fontsize=11,
                         color='black', fontweight='bold')
        if b > 0:
            ax_left.text(x_pos[i], c + b / 2, f'{b}',
                         ha='center', va='center', fontsize=11,
                         color='white', fontweight='bold')

    # 頂端標總數
    totals_left = c_vals + b_vals
    y_top_left = max(totals_left.max(), 1) * 1.15
    for i, t in enumerate(totals_left):
        ax_left.text(x_pos[i], t + y_top_left * 0.02, f'{t}',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels(algos, fontsize=11)
    ax_left.set_ylabel('Unassigned Urgent Customers', fontsize=12)
    ax_left.set_title('(a) TW-tight Failures',
                      fontsize=13, fontweight='bold')
    ax_left.set_ylim(0, y_top_left)
    ax_left.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax_left.grid(True, axis='y', alpha=0.3)
    ax_left.set_axisbelow(True)

    # === 右圖: Resource shortage (D) =============================
    d_vals = np.array([per_algo_subcat[a]['D_resource'] for a in algos])
    ax_right.bar(x_pos, d_vals, width,
                 color=COLOR_D_RESOURCE, edgecolor='black', linewidth=0.6,
                 label='D: Resource shortage (any vehicle feasible)')

    y_top_right = max(d_vals.max(), 1) * 1.15
    for i, d in enumerate(d_vals):
        if d > 0:
            ax_right.text(x_pos[i], d / 2, f'{d}',
                          ha='center', va='center', fontsize=11,
                          color='white', fontweight='bold')
        ax_right.text(x_pos[i], d + y_top_right * 0.02, f'{d}',
                      ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels(algos, fontsize=11)
    ax_right.set_ylabel('Unassigned Urgent Customers', fontsize=12)
    ax_right.set_title('(b) Resource-shortage Failures',
                       fontsize=13, fontweight='bold')
    ax_right.set_ylim(0, y_top_right)
    ax_right.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax_right.grid(True, axis='y', alpha=0.3)
    ax_right.set_axisbelow(True)

    fig.suptitle(
        f'Unassigned-Urgent Diagnosis  '
        f'(N={N_REQUESTS}, seed={SEED}, fleet 3 SLOW + 2 FAST + 1 UAV)',
        fontsize=13.5, fontweight='bold', y=1.00,
    )

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
