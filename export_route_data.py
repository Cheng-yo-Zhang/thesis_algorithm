"""
匯出 NN / TCGI / TCGI-ALNS 在 N=60 的演算法彙總指標 → CSV
=========================================================
不修改 exp_route_comparison.py；重用其 run_single_experiment / get_metrics，
確保 CSV 數字與圖表 (fig_*.png) 完全一致。

輸出: results/route_comparison/algo_summary.csv  (3 列, 對應 3 個演算法, N=60)
"""
import csv
from pathlib import Path

from exp_route_comparison import run_single_experiment, get_metrics

# ---- 參數 (與 exp_route_comparison 預設一致) ----
N_REQUESTS = 60
SEED = 42
ALGO_NAMES = ['NN', 'TCGI', 'TCGI-ALNS']

# ---- CSV 欄位 ----
COLUMNS = [
    'algorithm',        # 演算法名稱
    'n_requests',       # 請求總數 (= 60)
    'served',           # 實際完成數
    'missed',           # 未完成數
    'total',            # served + missed (= n_requests, 對帳用)
    'coverage',         # 服務率 (%)
    'last_departure',   # 最後完成時間 (min)
    'total_cost',       # 目標函數成本
    'total_distance',   # 車隊總距離 (km)
    'mcs_distance',     # MCS 總距離 (slow + fast)
    'uav_distance',     # UAV 總距離
    'slow_distance',    # MCS-SLOW 距離
    'fast_distance',    # MCS-FAST 距離
    'mcs_served',       # MCS 服務數
    'uav_served',       # UAV 服務數
]


def main():
    out_dir = Path("results") / "route_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "algo_summary.csv"

    print(f"Running experiment (N={N_REQUESTS}, seed={SEED}) ... (ALNS 5000 iter, 約 1-2 分鐘)")
    solutions, _ = run_single_experiment(N_REQUESTS, SEED)

    rows = []
    for name in ALGO_NAMES:
        m = get_metrics(solutions[name])
        rows.append({
            'algorithm':      name,
            'n_requests':     N_REQUESTS,
            'served':         m['served'],
            'missed':         m['missed'],
            'total':          m['total'],
            'coverage':       round(m['coverage'], 2),
            'last_departure': round(m['last_dep'], 2),
            'total_cost':     round(m['cost'], 2),
            'total_distance': round(m['distance'], 2),
            'mcs_distance':   round(m['mcs_distance'], 2),
            'uav_distance':   round(m['uav_distance'], 2),
            'slow_distance':  round(m['slow_distance'], 2),
            'fast_distance':  round(m['fast_distance'], 2),
            'mcs_served':     m['mcs_served'],
            'uav_served':     m['uav_served'],
        })

    # utf-8-sig: 讓 Windows Excel 直接正確開啟
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Saved] {out_path}\n")

    # 終端預覽
    print(f"{'algorithm':>10} {'served':>6} {'cov%':>6} {'lastDep':>8} "
          f"{'totDist':>8} {'mcsD':>7} {'uavD':>7} {'cost':>9}")
    for r in rows:
        print(f"{r['algorithm']:>10} {r['served']:>6} {r['coverage']:>6.1f} "
              f"{r['last_departure']:>8.1f} {r['total_distance']:>8.1f} "
              f"{r['mcs_distance']:>7.1f} {r['uav_distance']:>7.1f} {r['total_cost']:>9.1f}")


if __name__ == "__main__":
    main()
