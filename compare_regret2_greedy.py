"""
三策略 Construction Heuristic 比較: EDF vs Slack vs Regret-2
============================================================
產出：
  1. 並排路線圖 (指定 slots, 1×3 layout)
  2. Fleet Sizing 曲線 (service rate vs fleet size)
  3. 數值比較表
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from typing import List, Dict

from config import Config
from main import run_simulation


OUT_DIR = Path("comparison_3strategies")
STRATEGIES = ["edf", "slack", "regret2"]
STRATEGY_LABELS = {"edf": "Greedy-EDF", "slack": "Greedy-Slack", "regret2": "Regret-2"}
STRATEGY_COLORS = {"edf": "#9E9E9E", "slack": "#2196F3", "regret2": "#FF5722"}
STRATEGY_MARKERS = {"edf": "^", "slack": "o", "regret2": "s"}


# =====================================================================
# 1. 並排路線圖 (1×3)
# =====================================================================
def plot_side_by_side_routes(
    slot_data_map: Dict[str, List[dict]],
    cfg: Config,
    slots_to_compare: List[int],
    out_dir: Path,
):
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def vid_color(vid: int) -> str:
        return color_cycle[vid % len(color_cycle)]

    depot_x, depot_y = cfg.DEPOT_X, cfg.DEPOT_Y

    # 建 slot 索引
    slot_maps = {}
    for strat, sd_list in slot_data_map.items():
        slot_maps[strat] = {sd["slot"]: sd for sd in sd_list}

    for k in slots_to_compare:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))

        for ax, strat in zip(axes, STRATEGIES):
            smap = slot_maps[strat]
            if k not in smap:
                continue
            sd = smap[k]
            sol = sd["solution"]
            label = STRATEGY_LABELS[strat]

            # Depot
            ax.scatter(depot_x, depot_y, c="green", marker="s", s=180,
                       edgecolors="k", linewidths=1.2, zorder=10)
            ax.annotate("Depot", (depot_x, depot_y), fontsize=7, fontweight="bold",
                        ha="center", va="bottom", xytext=(0, 6), textcoords="offset points")

            route_handles = []

            if sol is not None:
                # MCS Routes
                for route in sol.mcs_routes:
                    if len(route.nodes) == 0:
                        continue
                    vid = route.vehicle_id
                    color = vid_color(vid)

                    start = route.start_position
                    sx = start.x if start else depot_x
                    sy = start.y if start else depot_y
                    xs = [sx] + [n.x for n in route.nodes] + [depot_x]
                    ys = [sy] + [n.y for n in route.nodes] + [depot_y]

                    ax.plot(xs, ys, "-", color=color, linewidth=2, alpha=0.7, zorder=4)

                    for seg in range(len(xs) - 1):
                        mx = (xs[seg] + xs[seg + 1]) / 2
                        my = (ys[seg] + ys[seg + 1]) / 2
                        ddx = xs[seg + 1] - xs[seg]
                        ddy = ys[seg + 1] - ys[seg]
                        if abs(ddx) > 0.01 or abs(ddy) > 0.01:
                            ax.annotate("", xy=(mx + ddx * 0.01, my + ddy * 0.01),
                                        xytext=(mx, my),
                                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

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

                    for seq_i, node in enumerate(route.nodes):
                        ax.annotate(f"{seq_i+1}", (node.x, node.y), fontsize=5,
                                    color=color, ha="left", va="top",
                                    xytext=(4, -4), textcoords="offset points")

                    vtype = route.vehicle_type.upper().replace("_", "-")
                    lbl = f"{vtype}-{vid}: {route.get_node_ids()}"
                    route_handles.append(Line2D([0], [0], color=color, linewidth=2, label=lbl))

                # UAV Routes
                for route in sol.uav_routes:
                    if len(route.nodes) == 0:
                        continue
                    vid = route.vehicle_id
                    color = vid_color(vid)

                    start = route.start_position
                    sx = start.x if start else depot_x
                    sy = start.y if start else depot_y
                    xs = [sx] + [n.x for n in route.nodes] + [depot_x]
                    ys = [sy] + [n.y for n in route.nodes] + [depot_y]

                    ax.plot(xs, ys, "--", color=color, linewidth=1.8, alpha=0.7, zorder=4)

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

                    for seq_i, node in enumerate(route.nodes):
                        ax.annotate(f"{seq_i+1}", (node.x, node.y), fontsize=5,
                                    color=color, ha="left", va="top",
                                    xytext=(4, -4), textcoords="offset points")

                    lbl = f"UAV-{vid}: {route.get_node_ids()}"
                    route_handles.append(Line2D([0], [0], color=color, linewidth=1.8,
                                                linestyle="--", label=lbl))

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

            n_assigned = sd["n_assigned"]
            n_unassigned = len(sol.unassigned_nodes) if sol else 0
            n_vehicles = sum(1 for r in sol.get_all_routes() if len(r.nodes) > 0) if sol else 0
            ax.set_title(
                f"{label}\n"
                f"Slot {k}  [t={sd['slot_start']:.0f}–{sd['slot_end']:.0f}]  "
                f"Assigned={n_assigned}  Unassigned={n_unassigned}  Vehicles={n_vehicles}",
                fontsize=10,
            )
            if route_handles:
                ax.legend(handles=route_handles, fontsize=6, loc="upper right")

        fig.tight_layout()
        path = out_dir / f"route_comparison_slot_{k:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {path}")


# =====================================================================
# 2. Fleet Sizing 曲線
# =====================================================================
def run_fleet_sizing_experiment(out_dir: Path):
    fleet_configs = [
        (20, 20, 20),
        (15, 15, 15),
        (10, 10, 10),
        (8,  8,  10),
        (6,  6,  8),
        (5,  5,  5),
        (4,  4,  4),
        (3,  3,  3),
    ]

    results: Dict[str, Dict[str, list]] = {s: {
        "fleet_label": [], "mw_rate": [], "mw_urgent_rate": [], "mw_normal_rate": [],
        "fleet_used": [], "mw_missed": [],
    } for s in STRATEGIES}

    for n_slow, n_fast, n_uav in fleet_configs:
        fleet_label = f"S{n_slow}/F{n_fast}/U{n_uav}"
        total = n_slow + n_fast + n_uav
        print(f"\n--- Fleet: {fleet_label} (total={total}) ---")

        for strategy in STRATEGIES:
            cfg = Config()
            cfg.CONSTRUCTION_STRATEGY = strategy
            cfg.NUM_MCS_SLOW = n_slow
            cfg.NUM_MCS_FAST = n_fast
            cfg.NUM_UAV = n_uav

            _, _, _, _, stats = run_simulation(cfg, verbose=False)

            results[strategy]["fleet_label"].append(fleet_label)
            results[strategy]["mw_rate"].append(stats["mw_service_rate"])
            results[strategy]["mw_urgent_rate"].append(stats["mw_urgent_rate"])
            results[strategy]["mw_normal_rate"].append(stats["mw_normal_rate"])
            results[strategy]["fleet_used"].append(stats["mw_fleet_used"])
            results[strategy]["mw_missed"].append(stats["mw_missed"])

            print(f"  {STRATEGY_LABELS[strategy]:>15s}: MW={stats['mw_service_rate']:.2%}  "
                  f"Urgent={stats['mw_urgent_rate']:.2%}  "
                  f"Normal={stats['mw_normal_rate']:.2%}  "
                  f"Used={stats['mw_fleet_used']}  Missed={stats['mw_missed']}")

    # --- 畫圖 ---
    x = np.arange(len(fleet_configs))
    labels = results["edf"]["fleet_label"]

    # Fig 1: MW Service Rate
    fig, ax = plt.subplots(figsize=(10, 5))
    for strat in STRATEGIES:
        ax.plot(x, [r * 100 for r in results[strat]["mw_rate"]],
                f"{STRATEGY_MARKERS[strat]}-", color=STRATEGY_COLORS[strat],
                linewidth=2, markersize=8, label=STRATEGY_LABELS[strat])
    ax.axhline(y=95, color="gray", linestyle="--", alpha=0.5, label="Target 95%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Fleet Configuration (Slow/Fast/UAV)", fontsize=11)
    ax.set_ylabel("MW Service Rate (%)", fontsize=11)
    ax.set_title("Fleet Sizing: Construction Heuristic Comparison — MW Service Rate", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    all_rates = []
    for s in STRATEGIES:
        all_rates.extend(results[s]["mw_rate"])
    ax.set_ylim(bottom=max(0, min(all_rates) * 100 - 5))
    fig.tight_layout()
    path = out_dir / "fleet_sizing_service_rate.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Saved] {path}")

    # Fig 2: Urgent Service Rate
    fig, ax = plt.subplots(figsize=(10, 5))
    for strat in STRATEGIES:
        ax.plot(x, [r * 100 for r in results[strat]["mw_urgent_rate"]],
                f"{STRATEGY_MARKERS[strat]}-", color=STRATEGY_COLORS[strat],
                linewidth=2, markersize=7, label=STRATEGY_LABELS[strat])
    ax.axhline(y=95, color="gray", linestyle="--", alpha=0.5, label="Target 95%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Fleet Configuration (Slow/Fast/UAV)", fontsize=11)
    ax.set_ylabel("Urgent Service Rate (%)", fontsize=11)
    ax.set_title("Fleet Sizing: Urgent Request Service Rate", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "fleet_sizing_urgent_rate.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

    # Fig 3: Fleet Used
    fig, ax = plt.subplots(figsize=(10, 5))
    n_strat = len(STRATEGIES)
    width = 0.8 / n_strat
    for i, strat in enumerate(STRATEGIES):
        offset = (i - (n_strat - 1) / 2) * width
        ax.bar(x + offset, results[strat]["fleet_used"], width,
               color=STRATEGY_COLORS[strat], alpha=0.8, label=STRATEGY_LABELS[strat])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Fleet Configuration", fontsize=11)
    ax.set_ylabel("Vehicles Used (MW)", fontsize=11)
    ax.set_title("Fleet Sizing: Actual Vehicles Used", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = out_dir / "fleet_sizing_vehicles_used.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

    return results


# =====================================================================
# 3. 數值比較表
# =====================================================================
def print_comparison_table(stats_map: Dict[str, dict]):
    print(f"\n{'='*80}")
    print(f"  Comparison: EDF vs Slack vs Regret-2")
    print(f"{'='*80}")

    header = f"{'Metric':<25} {'EDF':>12} {'Slack':>12} {'Regret-2':>12}"
    print(header)
    print("-" * len(header))

    e, s, r = stats_map["edf"], stats_map["slack"], stats_map["regret2"]

    rows = [
        ("MW Service Rate",   f"{e['mw_service_rate']:.2%}", f"{s['mw_service_rate']:.2%}", f"{r['mw_service_rate']:.2%}"),
        ("MW Urgent Rate",    f"{e['mw_urgent_rate']:.2%}",  f"{s['mw_urgent_rate']:.2%}",  f"{r['mw_urgent_rate']:.2%}"),
        ("MW Normal Rate",    f"{e['mw_normal_rate']:.2%}",  f"{s['mw_normal_rate']:.2%}",  f"{r['mw_normal_rate']:.2%}"),
        ("MW Served",         f"{e['mw_served']}",           f"{s['mw_served']}",           f"{r['mw_served']}"),
        ("MW Missed",         f"{e['mw_missed']}",           f"{s['mw_missed']}",           f"{r['mw_missed']}"),
        ("Fleet: Total Used", f"{e['mw_fleet_used']}",       f"{s['mw_fleet_used']}",       f"{r['mw_fleet_used']}"),
        ("Total Served",      f"{e['total_served']}",        f"{s['total_served']}",        f"{r['total_served']}"),
        ("Total Missed",      f"{e['total_missed']}",        f"{s['total_missed']}",        f"{r['total_missed']}"),
    ]

    for label, v_e, v_s, v_r in rows:
        print(f"{label:<25} {v_e:>12} {v_s:>12} {v_r:>12}")
    print(f"{'='*80}")


# =====================================================================
# Main
# =====================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Part 1: 基線比較 + 並排路線圖 ---
    print("\n" + "=" * 70)
    print("  Part 1: Baseline comparison + route maps")
    print("=" * 70)

    slot_data_map = {}
    stats_map = {}

    for strategy in STRATEGIES:
        print(f"\n  Running: {STRATEGY_LABELS[strategy]}")
        cfg = Config()
        cfg.CONSTRUCTION_STRATEGY = strategy

        slot_data, _, _, _, stats = run_simulation(cfg, verbose=False)
        slot_data_map[strategy] = slot_data
        stats_map[strategy] = stats

    print_comparison_table(stats_map)

    # 選取 MW 範圍內有代表性的 slots
    candidate_slots = []
    for sd in slot_data_map["edf"]:
        k = sd["slot"]
        if 49 <= k <= 100 and sd["solution"] is not None and sd["n_assigned"] >= 3:
            candidate_slots.append(k)
    slots_to_compare = candidate_slots[:5]

    print(f"\n  Plotting route comparisons for slots: {slots_to_compare}")
    cfg_plot = Config()
    plot_side_by_side_routes(slot_data_map, cfg_plot, slots_to_compare, OUT_DIR)

    # --- Part 2: Fleet Sizing 實驗 ---
    print("\n" + "=" * 70)
    print("  Part 2: Fleet Sizing experiment")
    print("=" * 70)

    run_fleet_sizing_experiment(OUT_DIR)

    print(f"\n  All outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
