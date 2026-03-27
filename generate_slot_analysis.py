"""
Generate detailed Slot 1–10 routing analysis in Markdown format.

Uses the with_uav config from compare_uav_flexibility.py.
Outputs to slot_analysis.md.
"""
import contextlib, io, math, sys
from copy import deepcopy
from typing import List, Dict, Tuple, Optional

from config import Config
from models import Node, VehicleState, Route, Solution
from problem import ChargingSchedulingProblem
from simulation import initialize_fleet, simulate_dispatch_window
from main import run_simulation


# ── helpers ──────────────────────────────────────────────────────────
def fmt(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}"

def manhattan(a, b) -> float:
    return abs(a.x - b.x) + abs(a.y - b.y)

def euclidean(a, b) -> float:
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def pos_str(n: Node) -> str:
    return f"({fmt(n.x)}, {fmt(n.y)})"

def is_depot(n: Node) -> bool:
    return n.node_type == 'depot'


def compute_slack(node: Node, dispatch_time: float, active_positions: list,
                  cfg: Config) -> float:
    max_power = max(cfg.MCS_FAST_POWER, cfg.MCS_SLOW_POWER)
    if active_positions:
        min_travel = min(
            (abs(node.x - p.x) + abs(node.y - p.y)) / cfg.MCS_SPEED
            for p in active_positions
        )
    else:
        min_travel = (abs(node.x - cfg.DEPOT_X) + abs(node.y - cfg.DEPOT_Y)) / cfg.MCS_SPEED
    min_service = (node.demand / max_power) * 60.0
    return node.due_date - (dispatch_time + min_travel + min_service)


def feasibility_from_depot(node: Node, depot: Node, start_time: float,
                           cfg: Config) -> dict:
    """Check feasibility for MCS-SLOW, MCS-FAST, UAV from depot."""
    results = {}

    for vtype, power, label in [
        ('mcs_slow', cfg.MCS_SLOW_POWER, 'MCS-SLOW'),
        ('mcs_fast', cfg.MCS_FAST_POWER, 'MCS-FAST'),
    ]:
        dist = manhattan(depot, node)
        travel = dist / cfg.MCS_SPEED
        arrival = start_time + travel
        setup, connect, disconnect = cfg.MCS_ARRIVAL_SETUP_MIN, cfg.MCS_CONNECT_MIN, cfg.MCS_DISCONNECT_MIN
        service_start = max(arrival + setup + connect, node.ready_time)
        charge_time = (node.demand / power) * 60.0
        departure = service_start + charge_time + disconnect
        feasible = departure <= node.due_date
        results[label] = {
            'dist': dist, 'dist_type': 'Manhattan',
            'travel': travel, 'arrival': arrival,
            'setup': setup, 'connect': connect,
            'service_start': service_start,
            'charge_time': charge_time, 'disconnect': disconnect,
            'departure': departure, 'due': node.due_date,
            'feasible': feasible, 'power': power,
        }

    # UAV
    dist = euclidean(depot, node)
    travel = dist / cfg.UAV_SPEED + cfg.UAV_TAKEOFF_LAND_OVERHEAD
    arrival = start_time + travel
    actual_delivery = min(node.demand, cfg.UAV_DELIVERABLE_ENERGY)
    charge_time = (actual_delivery / cfg.UAV_CHARGE_POWER) * 60.0
    departure = max(arrival, node.ready_time) + charge_time
    feasible = departure <= node.due_date
    results['UAV'] = {
        'dist': dist, 'dist_type': 'Euclidean',
        'travel': travel, 'arrival': arrival,
        'setup': 0, 'connect': 0,
        'service_start': max(arrival, node.ready_time),
        'charge_time': charge_time, 'disconnect': 0,
        'departure': departure, 'due': node.due_date,
        'feasible': feasible, 'power': cfg.UAV_CHARGE_POWER,
        'actual_delivery': actual_delivery,
    }
    return results


def find_assignment(node: Node, solution: Solution) -> Optional[Tuple[str, Route, int]]:
    """Find which route a node was assigned to."""
    for r in solution.mcs_routes:
        for i, n in enumerate(r.nodes):
            if n.id == node.id:
                vtype = r.vehicle_type.upper().replace('_', '-')
                return f"{vtype}-{r.vehicle_id}", r, i
    for r in solution.uav_routes:
        for i, n in enumerate(r.nodes):
            if n.id == node.id:
                return f"UAV-{r.vehicle_id}", r, i
    return None


# ── main output ──────────────────────────────────────────────────────
def generate_analysis():
    cfg = Config(
        RANDOM_SEED=42,
        URGENT_RATIO=0.5,
        CONSTRUCTION_STRATEGY="deadline",
        NUM_UAV=20,
        MCS_FAST_POWER=50.0,
        UAV_REQUEUE_FOR_MCS=False,
        PLOT_MAX_SLOTS=0,
        T_TOTAL=300,
        GENERATION_END=150,
        MEASUREMENT_START=0,
        MEASUREMENT_END=150,
    )

    # Run simulation and capture slot_data
    with contextlib.redirect_stdout(io.StringIO()):
        slot_data, fleet_final, problem, cfg_out, stats = run_simulation(cfg, verbose=True)

    # We need vehicle states at each slot boundary. Re-run step by step.
    problem2 = ChargingSchedulingProblem(cfg)
    fleet = initialize_fleet(cfg, problem2.depot)
    depot = problem2.depot
    all_requests: List[Node] = []
    backlog: List[Node] = []

    out = []  # collect lines
    def p(line=""):
        out.append(line)

    p("# Slot 1–10 詳細路徑規劃分析")
    p()
    p("## 實驗設定")
    p()
    p("| 參數 | 值 |")
    p("|------|-----|")
    p(f"| DELTA_T | {cfg.DELTA_T} min |")
    p(f"| MCS_SLOW_POWER | {cfg.MCS_SLOW_POWER} kW |")
    p(f"| MCS_FAST_POWER | {cfg.MCS_FAST_POWER} kW |")
    p(f"| UAV_CHARGE_POWER | {cfg.UAV_CHARGE_POWER} kW |")
    p(f"| MCS_SPEED | {cfg.MCS_SPEED} km/min ({cfg.MCS_SPEED*60:.1f} km/h) |")
    p(f"| UAV_SPEED | {cfg.UAV_SPEED} km/min ({cfg.UAV_SPEED*60:.1f} km/h) |")
    p(f"| MCS overhead | setup {cfg.MCS_ARRIVAL_SETUP_MIN} + connect {cfg.MCS_CONNECT_MIN} + disconnect {cfg.MCS_DISCONNECT_MIN} = {cfg.MCS_ARRIVAL_SETUP_MIN+cfg.MCS_CONNECT_MIN+cfg.MCS_DISCONNECT_MIN} min |")
    p(f"| UAV overhead | 起降 {cfg.UAV_TAKEOFF_LAND_OVERHEAD} min, 無 setup/connect |")
    p(f"| UAV_DELIVERABLE_ENERGY | {cfg.UAV_DELIVERABLE_ENERGY} kWh |")
    p(f"| Depot | ({cfg.DEPOT_X}, {cfg.DEPOT_Y}) |")
    p(f"| Fleet | MCS-SLOW×{cfg.NUM_MCS_SLOW}, MCS-FAST×{cfg.NUM_MCS_FAST}, UAV×{cfg.NUM_UAV} |")
    p()

    num_slots = cfg.T_TOTAL // cfg.DELTA_T
    max_analysis_slots = 10

    import numpy as np
    np.random.seed(cfg.RANDOM_SEED)
    import random
    random.seed(cfg.RANDOM_SEED)

    # Re-create problem fresh with same seed
    problem2 = ChargingSchedulingProblem(cfg)
    fleet = initialize_fleet(cfg, problem2.depot)
    depot = problem2.depot
    all_requests = []
    backlog = []

    for k in range(1, num_slots + 1):
        slot_start = (k - 1) * cfg.DELTA_T
        slot_end = k * cfg.DELTA_T
        dispatch_time = slot_end
        next_decision_time = slot_end + cfg.DELTA_T

        # Generate requests
        new_requests = problem2.generate_requests(slot_start, slot_number=k)

        # Process backlog
        active_backlog = []
        slot_missed = 0
        for req in backlog:
            if req.due_date <= dispatch_time:
                req.status = 'missed'
                slot_missed += 1
            else:
                req.status = 'backlog'
                active_backlog.append(req)

        active_pool = new_requests + active_backlog

        # Update committed nodes
        completed_committed = 0
        for vs in fleet:
            if vs.is_active:
                if vs.committed_nodes:
                    still_committed = []
                    still_departures = []
                    for node, dep_time in zip(vs.committed_nodes, vs.committed_departures):
                        if dep_time <= dispatch_time:
                            node.status = 'served'
                            completed_committed += 1
                        else:
                            still_committed.append(node)
                            still_departures.append(dep_time)
                    vs.committed_nodes = still_committed
                    vs.committed_departures = still_departures
                vs.available_time = max(vs.available_time, dispatch_time)
            else:
                vs.available_time = slot_end

        if k <= max_analysis_slots and active_pool:
            # ════════════════════════════════════════════════════
            #  OUTPUT ANALYSIS FOR THIS SLOT
            # ════════════════════════════════════════════════════
            p(f"---")
            p()
            p(f"## Slot {k}  [t={slot_start}–{slot_end} min]")
            p()

            # ── [1] Request Pool ──
            pool_sorted = sorted(active_pool, key=lambda n: (n.due_date, n.ready_time))
            n_urgent = sum(1 for n in new_requests if n.node_type == 'urgent')
            n_normal = len(new_requests) - n_urgent
            p(f"### 1. 請求池")
            p()
            p(f"新請求: {len(new_requests)} ({n_urgent} urgent, {n_normal} normal) ＋ Backlog: {len(active_backlog)}")
            p()
            p(f"| # | ID | 類型 | 位置 | 需求 (kWh) | ready | due | TW 寬度 | 來源 |")
            p(f"|---|-----|------|------|-----------|-------|-----|---------|------|")
            for idx, n in enumerate(pool_sorted, 1):
                src = "backlog" if n in active_backlog else "new"
                p(f"| {idx} | EV{n.id} | {n.node_type} | ({fmt(n.x)}, {fmt(n.y)}) | {fmt(n.demand)} | {fmt(n.ready_time)} | {fmt(n.due_date)} | {fmt(n.due_date - n.ready_time)} | {src} |")
            p()

            # ── [2] Active Vehicle States ──
            busy_vehicles = [vs for vs in fleet if vs.is_active and
                             (not is_depot(vs.position) or vs.available_time > dispatch_time or len(vs.committed_nodes) > 0)]
            p(f"### 2. 車輛狀態（非閒置）")
            p()
            if busy_vehicles:
                p(f"| 車輛 | 類型 | 位置 | available_time | committed | 說明 |")
                p(f"|------|------|------|---------------|-----------|------|")
                for vs in busy_vehicles:
                    vtype = vs.vehicle_type.upper().replace('_', '-')
                    comm_str = ", ".join(f"EV{n.id}" for n in vs.committed_nodes) if vs.committed_nodes else "—"
                    note = "正在服務" if vs.committed_nodes else "已部署"
                    p(f"| {vtype}-{vs.vehicle_id} | {vtype} | {pos_str(vs.position)} | {fmt(vs.available_time)} | {comm_str} | {note} |")
            else:
                p("所有車輛閒置於 depot，available_time = " + fmt(dispatch_time))
            p()
            p(f"dispatch_time = {dispatch_time}, next_decision_time = {next_decision_time}")
            p()

            # ── [3] Slack Computation ──
            active_positions = [vs.position for vs in fleet if vs.is_active]
            slack_dispatch = next_decision_time - cfg.DELTA_T  # used in construction
            max_power = max(cfg.MCS_FAST_POWER, cfg.MCS_SLOW_POWER)

            slack_data = []
            for n in active_pool:
                s = compute_slack(n, slack_dispatch, active_positions, cfg)
                slack_data.append((n, s))
            slack_data.sort(key=lambda x: (x[1], x[0].due_date))

            p(f"### 3. Slack 排序（Construction 處理順序）")
            p()
            p(f"```")
            p(f"slack = due_date − (dispatch_time + min_travel + min_service)")
            p(f"dispatch_time = {slack_dispatch}, max_power = {max_power} kW")
            p(f"```")
            p()
            p(f"| 處理順序 | ID | 類型 | slack (min) | due | 說明 |")
            p(f"|---------|-----|------|------------|-----|------|")
            for idx, (n, s) in enumerate(slack_data, 1):
                note = "⚠️ 負值 — MCS 可能不可行" if s < 0 else ("緊迫" if s < 30 else "寬裕")
                p(f"| {idx} | EV{n.id} | {n.node_type} | {fmt(s)} | {fmt(n.due_date)} | {note} |")
            p()

        # ── Run construction ──
        if not active_pool:
            if k <= max_analysis_slots:
                p(f"### 請求池為空，跳過建構")
                p()
            backlog = []
            continue

        all_customer_nodes = [n for n in all_requests if n.status not in ('served', 'missed')]
        all_customer_nodes.extend(new_requests)
        all_requests.extend(new_requests)
        problem2.setup_nodes(all_customer_nodes)

        solution = problem2.greedy_insertion_construction(active_pool, fleet,
                                                          dispatch_window_end=next_decision_time)

        n_assigned = sum(1 for n in active_pool if n.status == 'assigned')

        if k <= max_analysis_slots:
            # ── [4] Per-request insertion analysis ──
            p(f"### 4. 逐請求插入分析")
            p()

            for idx, (node, slk) in enumerate(slack_data, 1):
                assignment = find_assignment(node, solution)
                p(f"#### {idx}. EV{node.id} ({node.node_type}, due={fmt(node.due_date)})")
                p()

                # Feasibility from depot
                feas = feasibility_from_depot(node, depot, dispatch_time, cfg)

                for label in ['MCS-SLOW', 'MCS-FAST', 'UAV']:
                    f = feas[label]
                    status = "✅ 可行" if f['feasible'] else "❌ 不可行"
                    overhead_str = f"setup {f['setup']} + connect {f['connect']} + disconnect {f['disconnect']}" if label != 'UAV' else "無 overhead"

                    p(f"**{label} 從 depot（{f['dist_type']} {fmt(f['dist'])} km, {f['power']} kW）：** {status}")
                    p(f"```")
                    p(f"出發: {fmt(dispatch_time)} → 路程: {fmt(f['travel'])} min → 抵達: {fmt(f['arrival'])}")
                    if label != 'UAV':
                        p(f"overhead: {overhead_str} → 開始充電: {fmt(f['service_start'])}")
                    else:
                        p(f"開始充電: {fmt(f['service_start'])}  ({overhead_str})")
                    p(f"充電: ({fmt(node.demand if label != 'UAV' else f.get('actual_delivery', node.demand))}/{f['power']})×60 = {fmt(f['charge_time'])} min")
                    p(f"離開: {fmt(f['departure'])}  |  due: {fmt(f['due'])}  →  {'OK' if f['feasible'] else '超時 ' + fmt(f['departure'] - f['due']) + ' min'}")
                    p(f"```")
                    p()

                # Actual assignment
                if assignment:
                    vlabel, route, pos_in_route = assignment
                    p(f"**→ 指派: {vlabel}**")
                    # Show actual route timetable for this node
                    if pos_in_route < len(route.arrival_times):
                        arr = route.arrival_times[pos_in_route]
                        dep = route.departure_times[pos_in_route]
                        # Determine start position for this leg
                        if pos_in_route == 0:
                            start_pos = route.start_position if route.start_position else depot
                            start_label = "depot" if is_depot(start_pos) else f"EV{start_pos.id}"
                        else:
                            start_pos = route.nodes[pos_in_route - 1]
                            start_label = f"EV{start_pos.id}"
                        p(f"```")
                        p(f"從 {start_label} {pos_str(start_pos)} → EV{node.id} {pos_str(node)}")
                        p(f"抵達: {fmt(arr)} → 離開: {fmt(dep)}")
                        p(f"```")
                else:
                    if node in solution.unassigned_nodes:
                        p(f"**→ ❌ 未指派 (unassigned) → backlog**")
                    else:
                        p(f"**→ ❌ 未指派**")
                p()

            # ── [5] Route Summary ──
            p(f"### 5. 路徑結果")
            p()

            active_mcs = [r for r in solution.mcs_routes if len(r.nodes) > 0]
            active_uav = [r for r in solution.uav_routes if len(r.nodes) > 0]

            for r in active_mcs + active_uav:
                vtype = r.vehicle_type.upper().replace('_', '-')
                node_ids = ", ".join(str(n.id) for n in r.nodes)
                p(f"#### {vtype}-{r.vehicle_id}: [{node_ids}]")
                p()
                start_pos = r.start_position if r.start_position else depot
                start_label = "depot" if is_depot(start_pos) else f"EV{start_pos.id}"

                p(f"| Step | From | To | 距離 (km) | 路程 (min) | 抵達 | 開始充電 | 充電 (min) | 離開 |")
                p(f"|------|------|----|----------|-----------|------|---------|-----------|------|")

                prev = start_pos
                for i, node in enumerate(r.nodes):
                    from_label = start_label if i == 0 else f"EV{r.nodes[i-1].id}"
                    to_label = f"EV{node.id}"

                    if r.vehicle_type == 'uav':
                        dist = euclidean(prev, node)
                        travel = dist / cfg.UAV_SPEED + (cfg.UAV_TAKEOFF_LAND_OVERHEAD if prev.id != node.id else 0)
                    else:
                        dist = manhattan(prev, node)
                        travel = dist / cfg.MCS_SPEED

                    arr = r.arrival_times[i] if i < len(r.arrival_times) else 0
                    dep = r.departure_times[i] if i < len(r.departure_times) else 0

                    if r.vehicle_type == 'uav':
                        svc_start = max(arr, node.ready_time)
                        charge = dep - svc_start
                    else:
                        svc_start = max(arr + cfg.MCS_ARRIVAL_SETUP_MIN + cfg.MCS_CONNECT_MIN, node.ready_time)
                        charge = dep - svc_start - cfg.MCS_DISCONNECT_MIN

                    p(f"| {i+1} | {from_label} | {to_label} | {fmt(dist)} | {fmt(travel)} | {fmt(arr)} | {fmt(svc_start)} | {fmt(charge)} | {fmt(dep)} |")
                    prev = node

                # Return
                if r.vehicle_type == 'uav':
                    ret_dist = euclidean(prev, depot)
                else:
                    ret_dist = manhattan(prev, depot)
                p(f"| ↩ | {f'EV{prev.id}'} | depot | {fmt(ret_dist)} | — | — | — | — | — |")
                p()

            if not active_mcs and not active_uav:
                p("（無路徑）")
                p()

        # ── Simulate execution ──
        snapshot = deepcopy(solution)
        simulate_dispatch_window(solution, fleet, problem2, next_decision_time)

        # Build new backlog
        new_backlog = []
        for node in solution.unassigned_nodes:
            if node.due_date > next_decision_time:
                node.status = 'backlog'
                new_backlog.append(node)
            else:
                node.status = 'missed'
                slot_missed += 1

        for route in solution.get_all_routes():
            for node in route.nodes:
                if node.status == 'assigned':
                    node.status = 'backlog'
                    new_backlog.append(node)

        # UAV requeue (disabled in this config)
        backlog = new_backlog

        # Count served
        direct_served = sum(1 for r in solution.get_all_routes() for n in r.nodes if n.status == 'served')
        slot_served = completed_committed + direct_served

        if k <= max_analysis_slots and active_pool:
            # ── [6] Simulation Result ──
            p(f"### 6. 模擬執行結果（execution_end = {next_decision_time} min）")
            p()

            p(f"| 車輛 | 節點 | 抵達 | 離開 | 在 t={next_decision_time} 前完成？ | 狀態 |")
            p(f"|------|------|------|------|--------------|------|")

            for r in snapshot.mcs_routes + snapshot.uav_routes:
                if not r.nodes:
                    continue
                vtype = r.vehicle_type.upper().replace('_', '-')
                vlabel = f"{vtype}-{r.vehicle_id}"
                for i, node in enumerate(r.nodes):
                    arr = r.arrival_times[i] if i < len(r.arrival_times) else 0
                    dep = r.departure_times[i] if i < len(r.departure_times) else 0
                    if dep <= next_decision_time:
                        status_str = "✅ served"
                        done = "是"
                    elif arr < next_decision_time:
                        status_str = "🔄 committed (in_service)"
                        done = "已抵達未完成"
                    else:
                        # Check if vehicle departed (prev departure < next_decision_time)
                        prev_dep = r.start_time if i == 0 else (r.departure_times[i-1] if i-1 < len(r.departure_times) else 0)
                        if prev_dep < next_decision_time:
                            status_str = "🔄 committed (en-route)"
                            done = "已出發未抵達"
                        else:
                            status_str = "→ **backlog**"
                            done = "尚未出發"
                    p(f"| {vlabel} | EV{node.id} | {fmt(arr)} | {fmt(dep)} | {done} | {status_str} |")

            p()
            p(f"**Slot {k} 小結:** Assigned={n_assigned}  Served={slot_served}  Missed={slot_missed}  Backlog→{len(new_backlog)}")
            if new_backlog:
                bl_str = ", ".join(f"EV{n.id}(due={fmt(n.due_date)})" for n in new_backlog)
                p(f"  Backlog 內容: {bl_str}")
            p()

    # Write output
    with open("slot_analysis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Done: slot_analysis.md ({len(out)} lines)")


if __name__ == "__main__":
    generate_analysis()
