"""
Visualization & Reporting Module
=================================
- CSV export
- Per-slot EV spatial distribution figures (one file per slot)
- Per-slot vehicle route maps with MCS + UAV trajectories (one file per slot)
- Terminal detailed report
"""

import csv
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import Config
from problem import ChargingSchedulingProblem


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
    print(f"[Saved] {path}")


# ================================================================
#  Figure 1 — Per-Slot EV Spatial Distribution (one file per slot)
# ================================================================
def plot_per_slot_distribution(slot_data: List[dict], cfg: Config, out_dir: Path) -> None:
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

    slots_to_plot = slot_data[:cfg.PLOT_MAX_SLOTS] if cfg.PLOT_MAX_SLOTS > 0 else []
    for sd in slots_to_plot:
        k = sd["slot"]
        reqs = sd["new_requests"]
        backlog_nodes = sd["active_backlog"]

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

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
        ax.set_xlabel("X (km)", fontsize=9)
        ax.set_ylabel("Y (km)", fontsize=9)

        r_t = cfg.get_arrival_rate_profile(sd["slot_start"])
        ax.set_title(
            f"Slot {k}  [t={sd['slot_start']:.0f}–{sd['slot_end']:.0f} min]  "
            f"r(t)={r_t:.1f}\n"
            f"New: {len(reqs)} (U={n_urgent}, N={n_normal})  "
            f"Backlog: {len(backlog_nodes)}",
            fontsize=11,
        )

        fig.legend(handles=legend_elems, loc="lower center", ncol=4, fontsize=9,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=[0, 0.04, 1, 1])

        path = out_dir / f"fig1_slot_{k:02d}_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {path}")


# ================================================================
#  Figure 2 — Per-Slot Vehicle Route Map: MCS + UAV (one file per slot)
# ================================================================
def plot_per_slot_mcs_routes(slot_data: List[dict], cfg: Config,
                              problem: ChargingSchedulingProblem, out_dir: Path) -> None:
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def vid_color(vid: int) -> str:
        return color_cycle[vid % len(color_cycle)]

    depot_x, depot_y = cfg.DEPOT_X, cfg.DEPOT_Y

    slots_to_plot = slot_data[:cfg.PLOT_MAX_SLOTS] if cfg.PLOT_MAX_SLOTS > 0 else []
    for sd in slots_to_plot:
        k = sd["slot"]
        sol = sd["solution"]

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

        # Depot
        ax.scatter(depot_x, depot_y, c="green", marker="s", s=180,
                   edgecolors="k", linewidths=1.2, zorder=10)
        ax.annotate("Depot", (depot_x, depot_y), fontsize=7, fontweight="bold",
                    ha="center", va="bottom", xytext=(0, 6), textcoords="offset points")

        route_handles = []

        has_mcs = sol is not None and len(sol.mcs_routes) > 0
        has_uav = sol is not None and len(sol.uav_routes) > 0

        if not has_mcs and not has_uav:
            ax.text(cfg.AREA_SIZE / 2, cfg.AREA_SIZE / 2, "No routes",
                    ha="center", va="center", fontsize=12, color="gray")
        else:
            # ---------- MCS Routes (solid lines) ----------
            if has_mcs:
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

                    # 節點序列標注
                    for seq_i, node in enumerate(route.nodes):
                        ax.annotate(f"{seq_i+1}", (node.x, node.y), fontsize=5,
                                    color=color, ha="left", va="top",
                                    xytext=(4, -4), textcoords="offset points")

                    vtype = route.vehicle_type.upper().replace("_", "-")
                    if route.vehicle_type == 'mcs_slow':
                        type_idx = vid + 1
                    else:
                        type_idx = vid - cfg.NUM_MCS_SLOW + 1
                    label = f"{vtype}-{type_idx}: {route.get_node_ids()}"
                    route_handles.append(
                        Line2D([0], [0], color=color, linewidth=2, label=label)
                    )

            # ---------- UAV Routes (dashed lines) ----------
            if has_uav:
                for route in sol.uav_routes:
                    if len(route.nodes) == 0:
                        continue

                    vid = route.vehicle_id
                    color = vid_color(vid)

                    start = route.start_position if route.start_position else problem.depot
                    xs = [start.x] + [n.x for n in route.nodes] + [depot_x]
                    ys = [start.y] + [n.y for n in route.nodes] + [depot_y]

                    ax.plot(xs, ys, "--", color=color, linewidth=1.8, alpha=0.7, zorder=4)

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
                                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
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

                    # 節點序列標注
                    for seq_i, node in enumerate(route.nodes):
                        ax.annotate(f"{seq_i+1}", (node.x, node.y), fontsize=5,
                                    color=color, ha="left", va="top",
                                    xytext=(4, -4), textcoords="offset points")

                    type_idx = vid - cfg.NUM_MCS_SLOW - cfg.NUM_MCS_FAST + 1
                    label = f"UAV-{type_idx}: {route.get_node_ids()}"
                    route_handles.append(
                        Line2D([0], [0], color=color, linewidth=1.8,
                               linestyle="--", label=label)
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
        ax.set_xlabel("X (km)", fontsize=9)
        ax.set_ylabel("Y (km)", fontsize=9)

        n_served = sd["slot_served"]
        n_missed = sd["slot_missed"]
        n_backlog_out = len(sd["backlog_out"])
        ax.set_title(
            f"Slot {k}  [t={sd['slot_start']:.0f}–{sd['slot_end']:.0f} min]\n"
            f"Assigned={sd['n_assigned']}  Served={n_served}  "
            f"Missed={n_missed}  Backlog→{n_backlog_out}",
            fontsize=11,
        )
        if route_handles:
            ax.legend(handles=route_handles, fontsize=7, loc="upper right")

        fig.tight_layout()
        path = out_dir / f"fig2_slot_{k:02d}_routes.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {path}")


# ================================================================
#  終端機輸出
# ================================================================
def print_terminal_report(slot_data: List[dict], cfg: Config, stats: dict) -> None:
    sep = "=" * 100
    thin = "-" * 100

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
        bl = sd["active_backlog"]
        r_t = cfg.get_arrival_rate_profile(s)

        print(f"\n{sep}")
        print(f"  Slot {k}  [t={s:.0f} ~ {e:.0f} min]")
        print(sep)

        # === [1] Demand Generation ===
        n_urgent = sum(1 for n in reqs if n.node_type == "urgent")
        n_normal = len(reqs) - n_urgent
        print(f"\n  [1] Demand Generation: {len(reqs)} new requests "
              f"({n_urgent} urgent, {n_normal} normal)")

        # === [2] Request Pool (sorted by due_date) ===
        pool = sorted(reqs + bl, key=lambda n: (n.due_date, n.ready_time))
        print(f"\n  [2] Request Pool -- sorted by due_date (shortest deadline first):")
        if pool:
            print(f"      {'#':>3s}  {'EV':>5s}  {'Type':>7s}  {'Due_Date':>9s}  "
                  f"{'Demand':>10s}  {'Location':>14s}  {'Source':>7s}")
            print(f"      {thin[:72]}")
            for idx, n in enumerate(pool, 1):
                src = "backlog" if n in bl else "new"
                print(f"      {idx:3d}  EV{n.id:<3d}  {n.node_type:>7s}  {n.due_date:9.2f}  "
                      f"{n.demand:7.2f} kWh  ({n.x:5.2f},{n.y:5.2f})  {src:>7s}")
        else:
            print("      (empty)")
        if bl:
            print(f"      (includes {len(bl)} from backlog)")

        # === [3] MCS / UAV Dispatching ===
        sol = sd["solution"]
        if sol is not None:
            active_mcs = [r for r in sol.mcs_routes if len(r.nodes) > 0]
            active_uav = [r for r in sol.uav_routes if len(r.nodes) > 0]

            print(f"\n  [3] Vehicle Dispatching:")

            if not active_mcs and not active_uav:
                print("      (no routes)")
            else:
                for r in active_mcs:
                    vtype = r.vehicle_type.upper().replace("_", "-")
                    print(f"\n      --- {vtype}-{r.vehicle_id} ---")
                    print(f"      {'Step':>4s}  {'From':>8s}  {'To':>8s}  "
                          f"{'Dist(km)':>9s}  {'Travel':>9s}  {'Arrive':>9s}  "
                          f"{'Charge':>9s}  {'Depart':>9s}  {'MCS_kWh':>9s}")
                    print(f"      {thin[:82]}")

                    # Replay energy tracking
                    start_node = r.start_position
                    is_depot_start = (start_node is None
                                      or start_node.node_type == 'depot')
                    energy = r.start_energy if r.start_energy > 0 else cfg.MCS_CAPACITY
                    prev = start_node

                    for i, node in enumerate(r.nodes):
                        # Distance (Manhattan for MCS)
                        if prev is not None:
                            dx = abs(prev.x - node.x)
                            dy = abs(prev.y - node.y)
                            dist = dx + dy
                        else:
                            dx = abs(cfg.DEPOT_X - node.x)
                            dy = abs(cfg.DEPOT_Y - node.y)
                            dist = dx + dy

                        travel_time = dist / cfg.MCS_SPEED
                        travel_energy = dist * cfg.MCS_ENERGY_CONSUMPTION
                        energy -= (travel_energy + node.demand)

                        arr = r.arrival_times[i] if i < len(r.arrival_times) else 0
                        dep = r.departure_times[i] if i < len(r.departure_times) else 0
                        charge_min = dep - arr - cfg.MCS_ARRIVAL_SETUP_MIN - cfg.MCS_CONNECT_MIN - cfg.MCS_DISCONNECT_MIN
                        if charge_min < 0:
                            charge_min = node.demand / (cfg.MCS_FAST_POWER if r.vehicle_type == 'mcs_fast' else cfg.MCS_SLOW_POWER) * 60

                        from_label = "Depot" if (i == 0 and is_depot_start) else f"EV{prev.id}" if prev else "Depot"
                        to_label = f"EV{node.id}"

                        print(f"      {i+1:4d}  {from_label:>8s}  {to_label:>8s}  "
                              f"{dist:9.2f}  {travel_time:7.2f} m  {arr:9.2f}  "
                              f"{charge_min:7.2f} m  {dep:9.2f}  {energy:9.2f}")

                        prev = node

                    # Return to depot distance
                    if prev is not None:
                        ret_dist = abs(prev.x - cfg.DEPOT_X) + abs(prev.y - cfg.DEPOT_Y)
                        print(f"      >> Return to Depot: {ret_dist:.2f} km")

                for r in active_uav:
                    print(f"\n      --- UAV-{r.vehicle_id} ---")
                    print(f"      {'Step':>4s}  {'From':>8s}  {'To':>8s}  "
                          f"{'Dist(km)':>9s}  {'Travel':>9s}  {'Arrive':>9s}  "
                          f"{'Charge':>9s}  {'Depart':>9s}")
                    print(f"      {thin[:73]}")

                    start_node = r.start_position
                    is_depot_start = (start_node is None
                                      or start_node.node_type == 'depot')
                    prev = start_node

                    for i, node in enumerate(r.nodes):
                        if prev is not None:
                            dx = prev.x - node.x
                            dy = prev.y - node.y
                        else:
                            dx = cfg.DEPOT_X - node.x
                            dy = cfg.DEPOT_Y - node.y
                        dist = (dx**2 + dy**2) ** 0.5

                        travel_time = dist / cfg.UAV_SPEED + cfg.UAV_TAKEOFF_LAND_OVERHEAD
                        arr = r.arrival_times[i] if i < len(r.arrival_times) else 0
                        dep = r.departure_times[i] if i < len(r.departure_times) else 0
                        charge_min = dep - arr if dep > arr else 0

                        from_label = "Depot" if (i == 0 and is_depot_start) else f"EV{prev.id}" if prev else "Depot"
                        to_label = f"EV{node.id}"

                        print(f"      {i+1:4d}  {from_label:>8s}  {to_label:>8s}  "
                              f"{dist:9.2f}  {travel_time:7.2f} m  {arr:9.2f}  "
                              f"{charge_min:7.2f} m  {dep:9.2f}")
                        prev = node

            # === [4] Backlog ===
            backlog_out = sd["backlog_out"]
            unassigned = sol.unassigned_nodes if sol else []
            print(f"\n  [4] Backlog: {len(backlog_out)} requests not served -> carried to next slot")
            if backlog_out:
                items = ", ".join(f"EV{n.id}(due={n.due_date:.1f})" for n in backlog_out)
                print(f"      {items}")
            if sd["slot_missed"] > 0:
                print(f"      Missed (overdue): {sd['slot_missed']}")

        else:
            print(f"\n  [3] Vehicle Dispatching: (no scheduling — empty pool)")
            print(f"\n  [4] Backlog: 0")

        # Slot Summary
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

    uav_served_count = sum(1 for n in stats.get("all_requests", []) if n.uav_served)
    print(f"  UAV Served:          {uav_served_count}")

    # 實際使用的車輛數（至少服務過一筆請求）
    all_reqs = stats.get("all_requests", [])
    used_by_type = {}
    for r in all_reqs:
        if r.served_by_vehicle_id >= 0 and r.served_by_vehicle_type:
            used_by_type.setdefault(r.served_by_vehicle_type, set()).add(r.served_by_vehicle_id)
    n_slow_used = len(used_by_type.get('mcs_slow', set()))
    n_fast_used = len(used_by_type.get('mcs_fast', set()))
    n_uav_used = len(used_by_type.get('uav', set()))
    total_used = n_slow_used + n_fast_used + n_uav_used
    print(f"  Fleet Used:          {total_used}  (Slow={n_slow_used}, Fast={n_fast_used}, UAV={n_uav_used})")

    # 測量窗口統計
    mw_rate = stats.get("mw_service_rate")
    if mw_rate is not None:
        print(f"\n  --- Measurement Window [t={cfg.MEASUREMENT_START}-{cfg.MEASUREMENT_END}] ---")
        print(f"  MW Requests:         {stats['mw_total']}")
        print(f"  MW Served:           {stats['mw_served']}")
        print(f"  MW Service Rate:     {stats['mw_service_rate']:.1%}")
        print(f"  MW Urgent Rate:      {stats['mw_urgent_rate']:.1%}  ({stats['mw_urgent_served']}/{stats['mw_urgent_total']})")
        print(f"  MW Normal Rate:      {stats['mw_normal_rate']:.1%}  ({stats['mw_normal_served']}/{stats['mw_normal_total']})")
        print(f"  MW Fleet Used:       {stats['mw_fleet_used']}  (Slow={stats['mw_n_slow']}, Fast={stats['mw_n_fast']}, UAV={stats['mw_n_uav']})")
    print(sep)


# ================================================================
#  Strategy Comparison Charts (IEEE-friendly style)
# ================================================================
def plot_strategy_comparison(csv_path: str, output_dir: Path) -> None:
    """讀取 strategy_comparison.csv，產出三張比較圖表"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = df["strategy"].unique().tolist()
    rho_values = sorted(df["rho"].unique())

    # 統一色彩
    colors = {"nearest": "#7f8c8d", "deadline": "#3498db", "alns": "#e74c3c"}
    labels = {"nearest": "Nearest", "deadline": "Deadline", "alns": "ALNS"}

    bar_width = 0.22
    x = np.arange(len(rho_values))

    # --- Fig A: Service Rate ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, strat in enumerate(strategies):
        subset = df[df["strategy"] == strat].sort_values("rho")
        offsets = x + (i - len(strategies) / 2 + 0.5) * bar_width
        ax.bar(offsets, subset["mean_rate"], bar_width,
               yerr=subset["std_rate"], capsize=3,
               color=colors.get(strat, "#999"), edgecolor="k", linewidth=0.5,
               label=labels.get(strat, strat))
    ax.set_xlabel("Urgent Ratio ($\\rho$)", fontsize=11)
    ax.set_ylabel("Service Rate", fontsize=11)
    ax.set_title("Service Rate by Construction Strategy", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.1f}" for r in rho_values])
    ax.set_ylim(bottom=min(0.95, df["mean_rate"].min() - 0.01))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "fig_strategy_service_rate.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

    # --- Fig B: Fleet Usage ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, strat in enumerate(strategies):
        subset = df[df["strategy"] == strat].sort_values("rho")
        offsets = x + (i - len(strategies) / 2 + 0.5) * bar_width
        ax.bar(offsets, subset["mean_fleet_total"], bar_width,
               yerr=subset["std_fleet_total"], capsize=3,
               color=colors.get(strat, "#999"), edgecolor="k", linewidth=0.5,
               label=labels.get(strat, strat))
    ax.set_xlabel("Urgent Ratio ($\\rho$)", fontsize=11)
    ax.set_ylabel("Fleet Size Used", fontsize=11)
    ax.set_title("Fleet Usage by Construction Strategy", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.1f}" for r in rho_values])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "fig_strategy_fleet_usage.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

    # --- Fig C: Urgent & Normal Rate (two subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, strat in enumerate(strategies):
        subset = df[df["strategy"] == strat].sort_values("rho")
        offsets = x + (i - len(strategies) / 2 + 0.5) * bar_width
        ax1.bar(offsets, subset["mean_urgent_rate"], bar_width,
                color=colors.get(strat, "#999"), edgecolor="k", linewidth=0.5,
                label=labels.get(strat, strat))
        ax2.bar(offsets, subset["mean_normal_rate"], bar_width,
                color=colors.get(strat, "#999"), edgecolor="k", linewidth=0.5,
                label=labels.get(strat, strat))

    for ax, title in [(ax1, "Urgent Service Rate"), (ax2, "Normal Service Rate")]:
        ax.set_xlabel("Urgent Ratio ($\\rho$)", fontsize=11)
        ax.set_ylabel("Service Rate", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:.1f}" for r in rho_values])
        ax.set_ylim(bottom=min(0.93, df[["mean_urgent_rate", "mean_normal_rate"]].min().min() - 0.01))
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "fig_strategy_urgent_normal.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")
