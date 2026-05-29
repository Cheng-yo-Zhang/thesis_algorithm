"""
匯出 per-event 明細 (每完成一位客戶一列) → CSV
=================================================
這是兩張圖 (fig_served_vs_completion_time / fig_served_vs_distance) 的「母表」:
  - 圖一 = (departure_time, serve_rank)
  - 圖二 = (serve_rank, fleet_cum_distance)
兩者都是本表依 departure_time 排序後的投影。

距離計算完全比照 exp_route_comparison 圖二:
  UAV 用歐式距離, MCS 用曼哈頓距離, 不含返回 depot 的腿。

不修改 exp_route_comparison.py; 重用其 run_single_experiment。
輸出: results/route_comparison/route_events.csv
"""
import csv
from pathlib import Path

from exp_route_comparison import run_single_experiment

# ---- 參數 (與 exp_route_comparison 預設一致) ----
N_REQUESTS = 60
SEED = 42
ALGO_NAMES = ['NN', 'TCGI', 'TCGI-ALNS']

COLUMNS = [
    # 身分 / 地點
    'algorithm', 'serve_rank', 'node_id', 'node_type', 'x', 'y',
    'demand', 'ready_time', 'due_date',
    # 派遣
    'vehicle_type', 'vehicle_id', 'route_position', 'predecessor_id',
    # 時間
    'arrival_time', 'departure_time', 'user_waiting_time',
    'mcs_waiting_time', 'charging_mode',
    # 距離
    'hop_distance', 'vehicle_cum_distance', 'fleet_cum_distance',
]


def _get(lst, i):
    """安全取值, 超出範圍回傳 None。"""
    return lst[i] if (lst is not None and i < len(lst)) else None


def _r(v, n=3):
    """四捨五入; None -> 空字串 (CSV 友善)。"""
    return round(v, n) if v is not None else ''


def build_rows(name, sol, depot):
    # ---- pass 1: 依路徑 (per-vehicle) 算 hop / 車輛累積 / 前驅 / 站序 ----
    events = []
    for r in sol.get_all_routes():
        if not r.nodes:
            continue
        prev = depot
        vt = r.vehicle_type
        v_cum = 0.0
        for i, node in enumerate(r.nodes):
            dx = abs(prev.x - node.x)
            dy = abs(prev.y - node.y)
            hop = (dx * dx + dy * dy) ** 0.5 if vt == 'uav' else dx + dy
            v_cum += hop
            events.append({
                'node': node,
                'vehicle_type': vt,
                'vehicle_id': r.vehicle_id,
                'route_position': i + 1,
                'predecessor_id': -1 if prev is depot else prev.id,
                'arrival': _get(r.arrival_times, i),
                'departure': _get(r.departure_times, i),
                'user_wait': _get(r.user_waiting_times, i),
                'mcs_wait': _get(r.mcs_waiting_times, i),
                'charging_mode': _get(r.charging_modes, i) or '',
                'hop': hop,
                'vehicle_cum': v_cum,
            })
            prev = node

    # ---- pass 2: 全車隊依 departure_time 排序 -> serve_rank / 車隊累積 ----
    events.sort(key=lambda e: (e['departure'] if e['departure'] is not None else 0.0))
    rows = []
    fleet_cum = 0.0
    for rank, e in enumerate(events, 1):
        fleet_cum += e['hop']
        node = e['node']
        rows.append({
            'algorithm': name,
            'serve_rank': rank,
            'node_id': node.id,
            'node_type': node.node_type,
            'x': _r(node.x), 'y': _r(node.y),
            'demand': _r(node.demand),
            'ready_time': _r(node.ready_time, 2),
            'due_date': _r(node.due_date, 2),
            'vehicle_type': e['vehicle_type'],
            'vehicle_id': e['vehicle_id'],
            'route_position': e['route_position'],
            'predecessor_id': e['predecessor_id'],
            'arrival_time': _r(e['arrival'], 2),
            'departure_time': _r(e['departure'], 2),
            'user_waiting_time': _r(e['user_wait'], 2),
            'mcs_waiting_time': _r(e['mcs_wait'], 2),
            'charging_mode': e['charging_mode'],
            'hop_distance': _r(e['hop']),
            'vehicle_cum_distance': _r(e['vehicle_cum']),
            'fleet_cum_distance': _r(fleet_cum),
        })
    return rows


def main():
    out_dir = Path("results") / "route_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "route_events.csv"

    print(f"Running experiment (N={N_REQUESTS}, seed={SEED}) ... (ALNS 5000 iter, 約 1-2 分鐘)")
    solutions, problem = run_single_experiment(N_REQUESTS, SEED)
    depot = problem.depot

    all_rows = []
    for name in ALGO_NAMES:
        all_rows.extend(build_rows(name, solutions[name], depot))

    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[Saved] {out_path}  ({len(all_rows)} rows)\n")

    # 終端預覽: 每個演算法前 5 列
    hdr = f"{'algo':>10} {'rank':>4} {'id':>4} {'veh':>9} {'arr':>7} {'dep':>7} {'hop':>6} {'fleetCum':>9}"
    print(hdr)
    for name in ALGO_NAMES:
        for row in [r for r in all_rows if r['algorithm'] == name][:5]:
            print(f"{row['algorithm']:>10} {row['serve_rank']:>4} {row['node_id']:>4} "
                  f"{row['vehicle_type']:>9} {row['arrival_time']:>7} {row['departure_time']:>7} "
                  f"{row['hop_distance']:>6} {row['fleet_cum_distance']:>9}")


if __name__ == "__main__":
    main()
