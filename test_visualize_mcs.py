"""
MCS 排程可視化測試腳本
=======================
產出:
  1. CSV — requests.csv  (所有請求: 位置、需求電量、時間窗、所屬 slot)
  2. Figure 1 — 每個 Slot 的 EV 空間分布 (6 subplots)
  3. Figure 2 — 每個 Slot 結束後 MCS 路線圖 (6 subplots)
  4. 終端機完整輸出
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from main import (
    Config, Node, VehicleState, Route, Solution,
    ChargingSchedulingProblem, initialize_fleet, simulate_dispatch_window,
)

OUTPUT_DIR = Path("visualization_output")


# ================================================================
#  模擬 + 蒐集
# ================================================================
def run_simulation(cfg: Config):
    """執行 rolling-horizon 批次調度，回傳逐 slot 資料。"""
    problem = ChargingSchedulingProblem(cfg)
    fleet = initialize_fleet(cfg, problem.depot)

    all_requests: List[Node] = []
    backlog: List[Node] = []
    slot_data: List[dict] = []

    total_generated = 0
    total_served = 0
    total_missed = 0
    num_slots = cfg.T_TOTAL // cfg.DELTA_T

    for k in range(1, num_slots + 1):
        slot_start = (k - 1) * cfg.DELTA_T
        slot_end = k * cfg.DELTA_T
        dispatch_time = slot_end
        next_decision_time = slot_end + cfg.DELTA_T

        # --- 生成新請求 ---
        new_requests = problem.generate_requests(slot_start)
        total_generated += len(new_requests)

        # --- 處理 backlog ---
        active_backlog: List[Node] = []
        slot_missed = 0
        for req in backlog:
            if req.due_date <= dispatch_time:
                req.status = "missed"
                total_missed += 1
                slot_missed += 1
            else:
                req.status = "backlog"
                active_backlog.append(req)

        active_pool = new_requests + active_backlog

        # --- 更新 committed ---
        completed_committed = 0
        for vs in fleet:
            if vs.is_active:
                if vs.committed_nodes:
                    still_c, still_d = [], []
                    for node, dep in zip(vs.committed_nodes, vs.committed_departures):
                        if dep <= dispatch_time:
                            node.status = "served"
                            completed_committed += 1
                        else:
                            still_c.append(node)
                            still_d.append(dep)
                    vs.committed_nodes = still_c
                    vs.committed_departures = still_d
                vs.available_time = max(vs.available_time, dispatch_time)
            else:
                vs.available_time = slot_end
        total_served += completed_committed

        if not active_pool:
            slot_data.append({
                "slot": k, "slot_start": slot_start, "slot_end": slot_end,
                "new_requests": [], "active_backlog": [],
                "active_pool": [], "solution": None,
                "n_assigned": 0, "slot_served": completed_committed,
                "slot_missed": slot_missed, "backlog_out": [],
            })
            continue

        # --- 距離矩陣 ---
        all_customer_nodes = [n for n in all_requests if n.status not in ("served", "missed")]
        all_customer_nodes.extend(new_requests)
        all_requests.extend(new_requests)
        problem.setup_nodes(all_customer_nodes)

        # --- 求解 ---
        solution = problem.greedy_insertion_construction(active_pool, fleet, dispatch_window_end=next_decision_time)
        snapshot = deepcopy(solution)
        n_assigned = sum(1 for n in active_pool if n.status == "assigned")

        # --- 模擬執行 ---
        simulate_dispatch_window(solution, fleet, problem, next_decision_time)

        # --- 更新 backlog ---
        new_backlog: List[Node] = []
        for node in solution.unassigned_nodes:
            if node.due_date > next_decision_time:
                node.status = "backlog"
                new_backlog.append(node)
            else:
                node.status = "missed"
                total_missed += 1
                slot_missed += 1
        for route in solution.get_all_routes():
            for node in route.nodes:
                if node.status == "assigned":
                    node.status = "backlog"
                    new_backlog.append(node)
        backlog = new_backlog

        direct_served = sum(
            1 for r in solution.get_all_routes() for n in r.nodes if n.status == "served"
        )
        total_served += direct_served

        slot_data.append({
            "slot": k, "slot_start": slot_start, "slot_end": slot_end,
            "new_requests": new_requests, "active_backlog": active_backlog,
            "active_pool": active_pool, "solution": snapshot,
            "n_assigned": n_assigned,
            "slot_served": completed_committed + direct_served,
            "slot_missed": slot_missed,
            "backlog_out": new_backlog,
        })

    # --- Terminal: sweep committed ---
    terminal_committed = 0
    for vs in fleet:
        for node in vs.committed_nodes:
            node.status = "served"
            terminal_committed += 1
        vs.committed_nodes = []
        vs.committed_departures = []
    total_served += terminal_committed

    stats = {
        "total_generated": total_generated,
        "total_served": total_served,
        "total_missed": total_missed,
        "backlog_remaining": len(backlog),
        "terminal_committed": terminal_committed,
    }
    return slot_data, fleet, problem, stats


# ================================================================
#  CSV 匯出
# ================================================================
def export_requests_csv(slot_data: List[dict], path: Path) -> None:
    """匯出所有請求到 CSV: slot, id, x, y, demand, type, ready_time, due_date, tw_width"""
    fieldnames = [
        "slot", "request_id", "node_type", "x", "y",
        "demand_kwh", "ready_time", "due_date", "tw_width",
    ]
    rows = []
    for sd in slot_data:
        for node in sd["new_requests"]:
            rows.append({
                "slot": sd["slot"],
                "request_id": node.id,
                "node_type": node.node_type,
                "x": round(node.x, 4),
                "y": round(node.y, 4),
                "demand_kwh": round(node.demand, 4),
                "ready_time": round(node.ready_time, 4),
                "due_date": round(node.due_date, 4),
                "tw_width": round(node.due_date - node.ready_time, 4),
            })

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ================================================================
#  Figure 1 — 每個 Slot 的 EV 空間分布 (2x3 subplots)
# ================================================================
def plot_per_slot_distribution(slot_data: List[dict], cfg: Config, out_dir: Path) -> None:
    num_slots = len(slot_data)
    n_cols = 3
    n_rows = (num_slots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx in range(num_slots, len(axes)):
        axes[idx].set_visible(False)

    for idx, sd in enumerate(slot_data):
        ax = axes[idx]
        k = sd["slot"]
        reqs = sd["new_requests"]
        backlog_nodes = sd["active_backlog"]

        # Depot
        ax.scatter(cfg.DEPOT_X, cfg.DEPOT_Y, c="green", marker="s", s=180,
                   edgecolors="k", linewidths=1.2, zorder=10)
        ax.annotate("Depot", (cfg.DEPOT_X, cfg.DEPOT_Y), fontsize=7,
                    fontweight="bold", ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points")

        # Backlog nodes (灰色三角)
        for node in backlog_nodes:
            ax.scatter(node.x, node.y, c="#aaaaaa", marker="^", s=40,
                       edgecolors="k", linewidths=0.3, zorder=2, alpha=0.6)
            ax.annotate(f"{node.id}", (node.x, node.y), fontsize=5,
                        color="gray", ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points")

        # 新請求
        n_urgent = 0
        n_normal = 0
        for node in reqs:
            is_u = node.node_type == "urgent"
            color = "#e74c3c" if is_u else "#3498db"
            marker = "*" if is_u else "o"
            size = 140 if is_u else 60
            ax.scatter(node.x, node.y, c=color, marker=marker, s=size,
                       edgecolors="k", linewidths=0.5, zorder=5)
            ax.annotate(f"EV{node.id}", (node.x, node.y), fontsize=6,
                        fontweight="bold", ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points")
            if is_u:
                n_urgent += 1
            else:
                n_normal += 1

        ax.set_xlim(-0.5, cfg.AREA_SIZE + 0.5)
        ax.set_ylim(-0.5, cfg.AREA_SIZE + 0.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("X (km)", fontsize=8)
        ax.set_ylabel("Y (km)", fontsize=8)

        r_t = cfg.get_arrival_rate_profile(sd["slot_start"])
        ax.set_title(
            f"Slot {k}  [t={sd['slot_start']:.0f}-{sd['slot_end']:.0f} min]  "
            f"r(t)={r_t:.1f}\n"
            f"New: {len(reqs)} (U={n_urgent}, N={n_normal})  "
            f"Backlog: {len(backlog_nodes)}",
            fontsize=9,
        )

    # 圖例
    legend_elems = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#e74c3c",
               markeredgecolor="k", markersize=12, label="Urgent (new)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
               markeredgecolor="k", markersize=8, label="Normal (new)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#aaaaaa",
               markeredgecolor="k", markersize=8, label="Backlog"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="green",
               markeredgecolor="k", markersize=10, label="Depot"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Per-Slot EV Request Spatial Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    path = out_dir / "fig1_slot_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


# ================================================================
#  Figure 2 — 每個 Slot 結束後 MCS 路線圖 (2x3 subplots)
# ================================================================
def plot_per_slot_mcs_routes(slot_data: List[dict], cfg: Config,
                             problem: ChargingSchedulingProblem, out_dir: Path) -> None:
    num_slots = len(slot_data)
    n_cols = 3
    n_rows = (num_slots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx in range(num_slots, len(axes)):
        axes[idx].set_visible(False)

    # 固定色盤（按 vehicle_id 對應）
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def vid_color(vid: int) -> str:
        return color_cycle[vid % len(color_cycle)]

    depot_x, depot_y = cfg.DEPOT_X, cfg.DEPOT_Y

    for idx, sd in enumerate(slot_data):
        ax = axes[idx]
        k = sd["slot"]
        sol = sd["solution"]

        # Depot
        ax.scatter(depot_x, depot_y, c="green", marker="s", s=180,
                   edgecolors="k", linewidths=1.2, zorder=10)
        ax.annotate("Depot", (depot_x, depot_y), fontsize=7, fontweight="bold",
                    ha="center", va="bottom", xytext=(0, 6), textcoords="offset points")

        route_handles = []

        if sol is None or len(sol.mcs_routes) == 0:
            ax.text(cfg.AREA_SIZE / 2, cfg.AREA_SIZE / 2, "No routes",
                    ha="center", va="center", fontsize=12, color="gray")
        else:
            for route in sol.mcs_routes:
                if len(route.nodes) == 0:
                    continue

                vid = route.vehicle_id
                color = vid_color(vid)

                # 路徑座標: start → nodes → depot
                start = route.start_position if route.start_position else problem.depot
                xs = [start.x] + [n.x for n in route.nodes] + [depot_x]
                ys = [start.y] + [n.y for n in route.nodes] + [depot_y]

                ax.plot(xs, ys, "-", color=color, linewidth=2, alpha=0.7, zorder=4)

                # 箭頭
                for seg in range(len(xs) - 1):
                    mx = (xs[seg] + xs[seg + 1]) / 2
                    my = (ys[seg] + ys[seg + 1]) / 2
                    ddx = xs[seg + 1] - xs[seg]
                    ddy = ys[seg + 1] - ys[seg]
                    if abs(ddx) > 0.01 or abs(ddy) > 0.01:
                        ax.annotate(
                            "", xy=(mx + ddx * 0.01, my + ddy * 0.01),
                            xytext=(mx, my),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                        )

                # 節點
                for node in route.nodes:
                    is_u = node.node_type == "urgent"
                    mk = "*" if is_u else "o"
                    sz = 140 if is_u else 60
                    fc = "#e74c3c" if is_u else "#3498db"
                    ax.scatter(node.x, node.y, c=fc, marker=mk, s=sz,
                               edgecolors="k", linewidths=0.5, zorder=6)
                    ax.annotate(f"EV{node.id}", (node.x, node.y), fontsize=6,
                                fontweight="bold", ha="center", va="bottom",
                                xytext=(0, 5), textcoords="offset points")

                # 節點序列標注 (在路徑上標序號)
                for seq_i, node in enumerate(route.nodes):
                    ax.annotate(f"{seq_i+1}", (node.x, node.y), fontsize=5,
                                color=color, ha="left", va="top",
                                xytext=(4, -4), textcoords="offset points")

                vtype = route.vehicle_type.upper().replace("_", "-")
                label = f"{vtype}-{vid}: {route.get_node_ids()}"
                route_handles.append(
                    Line2D([0], [0], color=color, linewidth=2, label=label)
                )

            # Unassigned
            if sol.unassigned_nodes:
                for node in sol.unassigned_nodes:
                    ax.scatter(node.x, node.y, c="none", marker="o", s=100,
                               edgecolors="red", linewidths=2, zorder=7, linestyle="--")
                    ax.annotate(f"EV{node.id}\n(unassigned)", (node.x, node.y),
                                fontsize=5, color="red", ha="center", va="top",
                                xytext=(0, -8), textcoords="offset points")

        ax.set_xlim(-0.5, cfg.AREA_SIZE + 0.5)
        ax.set_ylim(-0.5, cfg.AREA_SIZE + 0.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("X (km)", fontsize=8)
        ax.set_ylabel("Y (km)", fontsize=8)

        n_served = sd["slot_served"]
        n_missed = sd["slot_missed"]
        n_backlog_out = len(sd["backlog_out"])
        ax.set_title(
            f"Slot {k}  [t={sd['slot_start']:.0f}-{sd['slot_end']:.0f} min]\n"
            f"Assigned={sd['n_assigned']}  Served={n_served}  "
            f"Missed={n_missed}  Backlog→{n_backlog_out}",
            fontsize=9,
        )
        if route_handles:
            ax.legend(handles=route_handles, fontsize=6, loc="upper right")

    fig.suptitle("Per-Slot MCS Route Map (after scheduling)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "fig2_slot_mcs_routes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


# ================================================================
#  終端機輸出
# ================================================================
def print_terminal_report(slot_data: List[dict], cfg: Config, stats: dict) -> None:
    sep = "=" * 100
    thin = "-" * 100
    slot_sep = "-" * 80

    print(f"\n{sep}")
    print("  MCS ROLLING-HORIZON BATCH DISPATCH -- DETAILED REPORT")
    print(sep)
    print(f"  Config: T_TOTAL={cfg.T_TOTAL}min  DELTA_T={cfg.DELTA_T}min  "
          f"LAMBDA_BAR={cfg.LAMBDA_BAR}  URGENT_RATIO={cfg.URGENT_RATIO}")
    print(f"  Fleet:  S-MCS={cfg.NUM_MCS_SLOW}  F-MCS={cfg.NUM_MCS_FAST}  UAV={cfg.NUM_UAV}")
    print(f"  Area:   {cfg.AREA_SIZE} x {cfg.AREA_SIZE} km  Depot=({cfg.DEPOT_X}, {cfg.DEPOT_Y})")
    print(sep)

    # --- 逐 Slot 詳細 ---
    for sd in slot_data:
        k = sd["slot"]
        s, e = sd["slot_start"], sd["slot_end"]
        reqs = sd["new_requests"]
        r_t = cfg.get_arrival_rate_profile(s)

        print(f"\n{slot_sep}")
        print(f"  Slot {k}  [t={s:.0f} ~ {e:.0f} min]   r(t)={r_t:.1f}   "
              f"E[N]={cfg.LAMBDA_BAR * r_t * cfg.DELTA_T:.2f}")
        print(f"{slot_sep}")

        # 新請求表
        if reqs:
            print(f"\n  New Requests ({len(reqs)}):")
            header = (f"    {'ID':>4s}  {'Type':>7s}  {'Location':>14s}  "
                      f"{'ready_t':>8s}  {'due_date':>9s}  {'TW':>7s}  {'Demand':>10s}")
            print(header)
            print(f"    {thin[:75]}")
            for n in sorted(reqs, key=lambda x: x.ready_time):
                tw = n.due_date - n.ready_time
                print(f"    {n.id:4d}  {n.node_type:>7s}  ({n.x:6.2f},{n.y:5.2f})  "
                      f"{n.ready_time:8.2f}  {n.due_date:9.2f}  {tw:7.1f}  {n.demand:7.2f} kWh")
        else:
            print("\n  New Requests: 0")

        # Backlog
        bl = sd["active_backlog"]
        if bl:
            print(f"\n  Active Backlog ({len(bl)}): {[n.id for n in bl]}")

        # MCS 路線
        sol = sd["solution"]
        if sol is not None:
            active_routes = [r for r in sol.mcs_routes if len(r.nodes) > 0]
            print(f"\n  MCS Routes ({len(active_routes)} active / {len(sol.mcs_routes)} total):")
            for r in active_routes:
                vtype = r.vehicle_type.upper().replace("_", "-")
                ids = r.get_node_ids()

                # 建立節點詳細時刻表
                print(f"    {vtype}-{r.vehicle_id}:  {ids}")
                print(f"      dist={r.total_distance:.1f} km  "
                      f"time={r.total_time:.1f} min  "
                      f"demand={r.total_demand:.1f} kWh")

                if r.arrival_times and r.departure_times:
                    print(f"      {'Seq':>3s}  {'EV':>4s}  {'Type':>6s}  "
                          f"{'Arrive':>8s}  {'Depart':>8s}  {'Wait':>7s}  {'Mode':>5s}")
                    for i, node in enumerate(r.nodes):
                        arr = r.arrival_times[i] if i < len(r.arrival_times) else 0
                        dep = r.departure_times[i] if i < len(r.departure_times) else 0
                        uwt = r.user_waiting_times[i] if i < len(r.user_waiting_times) else 0
                        mode = r.charging_modes[i] if i < len(r.charging_modes) else "?"
                        tag = "U" if node.node_type == "urgent" else "N"
                        print(f"      {i+1:3d}  {node.id:4d}  {tag:>6s}  "
                              f"{arr:8.2f}  {dep:8.2f}  {uwt:7.2f}  {mode:>5s}")

            if sol.unassigned_nodes:
                print(f"\n  Unassigned: {[n.id for n in sol.unassigned_nodes]}")
        else:
            print("\n  (no scheduling — empty pool)")

        # Slot 統計摘要
        print(f"\n  Slot Summary: assigned={sd['n_assigned']}  served={sd['slot_served']}  "
              f"missed={sd['slot_missed']}  backlog_out={len(sd['backlog_out'])}")

    # --- 最終統計 ---
    print(f"\n{sep}")
    print("  SIMULATION COMPLETE -- FINAL STATISTICS")
    print(sep)
    g = stats["total_generated"]
    s = stats["total_served"]
    m = stats["total_missed"]
    b = stats["backlog_remaining"]
    tc = stats["terminal_committed"]
    print(f"  Total Generated:     {g}")
    print(f"  Total Served:        {s}" + (f"  (includes {tc} terminal-committed)" if tc else ""))
    print(f"  Total Missed:        {m}")
    print(f"  Remaining Backlog:   {b}")
    if g > 0:
        print(f"  Coverage Rate:       {s/g:.1%}")
        print(f"  Miss Rate:           {m/g:.1%}")
    acc = s + m + b
    if acc != g:
        print(f"  WARNING: accounting mismatch — accounted={acc}, generated={g}")
    print(sep)


# ================================================================
#  Main
# ================================================================
if __name__ == "__main__":
    cfg = Config()

    print("Running simulation ...")
    print(f"Config: T={cfg.T_TOTAL}min, dt={cfg.DELTA_T}min, "
          f"lambda={cfg.LAMBDA_BAR}, rho={cfg.URGENT_RATIO}")

    slot_data, fleet, problem, stats = run_simulation(cfg)

    # 輸出資料夾
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. CSV
    csv_path = OUTPUT_DIR / "requests.csv"
    export_requests_csv(slot_data, csv_path)
    print(f"[Saved] {csv_path}")

    # 2. 圖表
    plot_per_slot_distribution(slot_data, cfg, OUTPUT_DIR)
    plot_per_slot_mcs_routes(slot_data, cfg, problem, OUTPUT_DIR)

    # 3. 終端機報告
    print_terminal_report(slot_data, cfg, stats)
