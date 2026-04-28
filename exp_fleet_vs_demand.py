"""
仿 Qureshi et al. Fig. 6:
    x-axis: 充電請求數量 (n)
    y-axis: 所需車輛數量 (分 Slow / Fast / UAV 堆疊)
    三條線: Greedy (Deadline) | Regret-2 | Regret-2 + ALNS

靜態單次調度 — 生成 n 個請求，各演算法各解一次，統計使用車輛數。
"""

import numpy as np
import time
import csv
import json
from pathlib import Path
from copy import deepcopy

from config import Config
from models import Node, VehicleState, Route, Solution
from problem import ChargingSchedulingProblem
from simulation import initialize_fleet
from alns import ALNSSolver


# === Experiment Parameters ===
REQUEST_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80]
NUM_REPLICATIONS = 10
BASE_SEED = 42
OUTPUT_DIR = Path("results_fleet_vs_demand")

STRATEGIES = [
    {"name": "Greedy (EDF)",    "construction": "edf",     "alns_iter": 0},
    {"name": "Regret-2",        "construction": "regret2", "alns_iter": 0},
    {"name": "Regret-2 + ALNS", "construction": "regret2", "alns_iter": 1000},
]


def generate_static_requests(problem: ChargingSchedulingProblem, n: int,
                             cfg: Config) -> list:
    """生成 n 個靜態請求（模擬 t=0 時刻一次性到達）"""
    nodes = []
    for _ in range(n):
        node_id = problem._next_node_id
        problem._next_node_id += 1

        is_urgent = np.random.random() < cfg.URGENT_RATIO

        # 位置
        if is_urgent and cfg.URGENT_CORNER_RATIO > 0 and np.random.random() < cfg.URGENT_CORNER_RATIO:
            import random
            corner = random.choice(cfg.CORNER_POSITIONS)
            x = np.clip(np.random.normal(corner[0], cfg.URGENT_CORNER_SIGMA), 0, cfg.AREA_SIZE)
            y = np.clip(np.random.normal(corner[1], cfg.URGENT_CORNER_SIGMA), 0, cfg.AREA_SIZE)
        else:
            x = np.random.uniform(0, cfg.AREA_SIZE)
            y = np.random.uniform(0, cfg.AREA_SIZE)

        if is_urgent:
            node_type = 'urgent'
            demand = problem._sample_truncated_demand(cfg.URGENT_ENERGY_MIN, cfg.URGENT_ENERGY_MAX)
            tw_width = np.random.uniform(cfg.URGENT_TW_MIN, cfg.URGENT_TW_MAX)
            ev_soc = np.random.uniform(cfg.URGENT_SOC_MIN, cfg.URGENT_SOC_MAX)
        else:
            node_type = 'normal'
            demand = problem._sample_truncated_demand(cfg.NORMAL_ENERGY_MIN, cfg.NORMAL_ENERGY_MAX)
            tw_width = np.random.uniform(cfg.NORMAL_TW_MIN, cfg.NORMAL_TW_MAX)
            ev_soc = max(0.05, 1.0 - demand / cfg.EV_BATTERY_CAPACITY)

        t_i = np.random.uniform(0, cfg.DELTA_T)
        d_i = t_i + tw_width

        node = Node(
            id=node_id, x=x, y=y, demand=demand,
            ready_time=t_i, due_date=d_i, service_time=0,
            node_type=node_type, status='new',
            ev_current_soc=ev_soc,
        )
        nodes.append(node)
    return nodes


def solve_once(requests, fleet, problem, cfg, strategy):
    """單次調度求解，回傳 solution"""
    if strategy["construction"] == "regret2":
        solution = problem.regret2_insertion_construction(requests, fleet)
    else:
        solution = problem.greedy_insertion_construction(requests, fleet)

    assigned_count = sum(len(r.nodes) for r in solution.get_all_routes())
    if assigned_count > 0 and strategy["alns_iter"] > 0:
        solver = ALNSSolver(problem, cfg)
        solution = solver.solve(solution)

    return solution


def count_vehicles_used(solution):
    """統計各車型使用的車輛數（route 有節點才算）"""
    n_slow, n_fast, n_uav = 0, 0, 0
    for route in solution.mcs_routes:
        if len(route.nodes) > 0:
            if route.vehicle_type == 'mcs_slow':
                n_slow += 1
            elif route.vehicle_type == 'mcs_fast':
                n_fast += 1
    for route in solution.uav_routes:
        if len(route.nodes) > 0:
            n_uav += 1
    return n_slow, n_fast, n_uav


def run_experiment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    results = []

    total_runs = len(REQUEST_COUNTS) * len(STRATEGIES) * NUM_REPLICATIONS
    run_count = 0

    print(f"{'='*80}")
    print(f"Fleet vs Demand (Qureshi Fig.6 Style)")
    print(f"{'='*80}")
    print(f"  Request counts: {REQUEST_COUNTS}")
    print(f"  Strategies: {[s['name'] for s in STRATEGIES]}")
    print(f"  Replications: {NUM_REPLICATIONS}")
    print(f"  Total runs: {total_runs}")
    print()

    for n_req in REQUEST_COUNTS:
        for strategy in STRATEGIES:
            rep_data = {
                "n_slow": [], "n_fast": [], "n_uav": [], "total": [],
                "service_rate": [], "urgent_rate": [], "normal_rate": [],
                "cost": [], "elapsed": [],
            }

            for rep in range(NUM_REPLICATIONS):
                seed = BASE_SEED + rep
                np.random.seed(seed)
                import random
                random.seed(seed)

                cfg = Config(
                    RANDOM_SEED=seed,
                    CONSTRUCTION_STRATEGY=strategy["construction"],
                    ALNS_MAX_ITERATIONS=strategy["alns_iter"],
                    NUM_MCS_SLOW=5,
                    NUM_MCS_FAST=5,
                    NUM_UAV=5,
                )
                problem = ChargingSchedulingProblem(cfg)
                fleet = initialize_fleet(cfg, problem.depot)

                # 生成靜態請求
                requests = generate_static_requests(problem, n_req, cfg)
                problem.setup_nodes(requests)

                # 深拷貝，讓每個策略用同一組請求
                req_copy = deepcopy(requests)
                fleet_copy = deepcopy(fleet)
                for r in req_copy:
                    r.status = 'new'

                # 求解 & 計時
                t0 = time.time()
                solution = solve_once(req_copy, fleet_copy, problem, cfg, strategy)
                elapsed = time.time() - t0

                # 統計
                n_slow, n_fast, n_uav = count_vehicles_used(solution)
                n_assigned = sum(len(r.nodes) for r in solution.get_all_routes())
                n_unassigned = len(solution.unassigned_nodes)
                sr = n_assigned / n_req if n_req > 0 else 1.0

                # Urgent / Normal
                urgent_total = sum(1 for r in req_copy if r.node_type == 'urgent')
                normal_total = sum(1 for r in req_copy if r.node_type == 'normal')
                urgent_assigned = sum(
                    1 for route in solution.get_all_routes()
                    for node in route.nodes if node.node_type == 'urgent'
                )
                normal_assigned = sum(
                    1 for route in solution.get_all_routes()
                    for node in route.nodes if node.node_type == 'normal'
                )
                ur = urgent_assigned / urgent_total if urgent_total > 0 else 1.0
                nr = normal_assigned / normal_total if normal_total > 0 else 1.0

                rep_data["n_slow"].append(n_slow)
                rep_data["n_fast"].append(n_fast)
                rep_data["n_uav"].append(n_uav)
                rep_data["total"].append(n_slow + n_fast + n_uav)
                rep_data["service_rate"].append(sr)
                rep_data["urgent_rate"].append(ur)
                rep_data["normal_rate"].append(nr)
                rep_data["cost"].append(solution.total_cost)
                rep_data["elapsed"].append(elapsed)

                run_count += 1

            row = {
                "n_requests": n_req,
                "strategy": strategy["name"],
                "mean_n_slow": np.mean(rep_data["n_slow"]),
                "mean_n_fast": np.mean(rep_data["n_fast"]),
                "mean_n_uav": np.mean(rep_data["n_uav"]),
                "mean_total": np.mean(rep_data["total"]),
                "std_total": np.std(rep_data["total"], ddof=1) if NUM_REPLICATIONS > 1 else 0,
                "mean_service_rate": np.mean(rep_data["service_rate"]),
                "mean_urgent_rate": np.mean(rep_data["urgent_rate"]),
                "mean_normal_rate": np.mean(rep_data["normal_rate"]),
                "mean_cost": np.mean(rep_data["cost"]),
                "mean_elapsed": np.mean(rep_data["elapsed"]),
            }
            results.append(row)

            elapsed_total = time.time() - t_start
            print(f"  [{run_count:3d}/{total_runs}] n={n_req:3d} {strategy['name']:>17s} | "
                  f"fleet: S={row['mean_n_slow']:.1f} F={row['mean_n_fast']:.1f} "
                  f"U={row['mean_n_uav']:.1f} T={row['mean_total']:.1f} | "
                  f"rate={row['mean_service_rate']:.3f} | "
                  f"time={row['mean_elapsed']:.2f}s [{elapsed_total:.0f}s]")

    # Save CSV
    csv_path = OUTPUT_DIR / "fleet_vs_demand.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Save JSON
    json_path = OUTPUT_DIR / "fleet_vs_demand.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    total_time = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    print(f"  {'n':>4} {'Strategy':>17} {'Slow':>5} {'Fast':>5} {'UAV':>5} "
          f"{'Total':>6} {'Rate':>6} {'Urg':>6} {'Nor':>6} {'Time':>7}")
    print(f"  {'-'*4} {'-'*17} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for r in results:
        print(f"  {r['n_requests']:4d} {r['strategy']:>17s} "
              f"{r['mean_n_slow']:5.1f} {r['mean_n_fast']:5.1f} {r['mean_n_uav']:5.1f} "
              f"{r['mean_total']:6.1f} {r['mean_service_rate']:6.3f} "
              f"{r['mean_urgent_rate']:6.3f} {r['mean_normal_rate']:6.3f} "
              f"{r['mean_elapsed']:7.2f}s")
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Results saved to {csv_path}")

    plot_results(results)
    return results


def plot_results(results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    colors = {
        "Greedy (EDF)": "#7f8c8d",
        "Regret-2": "#3498db",
        "Regret-2 + ALNS": "#e74c3c",
    }
    markers = {
        "Greedy (EDF)": "s",
        "Regret-2": "^",
        "Regret-2 + ALNS": "o",
    }
    type_colors = {"Slow": "#2ecc71", "Fast": "#e67e22", "UAV": "#9b59b6"}

    # === (a) Total vehicles used (line chart) ===
    ax = axes[0]
    for s in STRATEGIES:
        sname = s["name"]
        rows = [r for r in results if r["strategy"] == sname]
        ns = [r["n_requests"] for r in rows]
        totals = [r["mean_total"] for r in rows]
        ax.plot(ns, totals, label=sname, color=colors[sname],
                marker=markers[sname], linewidth=2, markersize=8)
    ax.set_xlabel("Number of Charging Requests", fontsize=12)
    ax.set_ylabel("Number of Vehicles Used", fontsize=12)
    ax.set_title("(a) Vehicles Required vs Demand", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # === (b) Stacked bar: vehicle type breakdown ===
    ax = axes[1]
    x_positions = np.arange(len(REQUEST_COUNTS))
    width = 0.25
    for i, s in enumerate(STRATEGIES):
        sname = s["name"]
        rows = [r for r in results if r["strategy"] == sname]
        slow = [r["mean_n_slow"] for r in rows]
        fast = [r["mean_n_fast"] for r in rows]
        uav = [r["mean_n_uav"] for r in rows]
        offset = (i - 1) * width
        ax.bar(x_positions + offset, slow, width, label="Slow" if i == 0 else "",
               color=type_colors["Slow"], edgecolor="white")
        ax.bar(x_positions + offset, fast, width, bottom=slow,
               label="Fast" if i == 0 else "",
               color=type_colors["Fast"], edgecolor="white")
        ax.bar(x_positions + offset, uav, width,
               bottom=[s + f for s, f in zip(slow, fast)],
               label="UAV" if i == 0 else "",
               color=type_colors["UAV"], edgecolor="white")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(REQUEST_COUNTS)
    ax.set_xlabel("Number of Charging Requests", fontsize=12)
    ax.set_ylabel("Number of Vehicles Used", fontsize=12)
    ax.set_title("(b) Fleet Composition by Algorithm", fontsize=13)
    # Strategy labels below bars
    for i, s in enumerate(STRATEGIES):
        for j, n in enumerate(REQUEST_COUNTS):
            if j == 0:
                ax.text(j + (i - 1) * width, -1.5, s["name"].split("(")[0].strip(),
                        ha='center', fontsize=6, rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # === (c) Service Rate ===
    ax = axes[2]
    for s in STRATEGIES:
        sname = s["name"]
        rows = [r for r in results if r["strategy"] == sname]
        ns = [r["n_requests"] for r in rows]
        rates = [r["mean_service_rate"] * 100 for r in rows]
        ax.plot(ns, rates, label=sname, color=colors[sname],
                marker=markers[sname], linewidth=2, markersize=8)
    ax.set_xlabel("Number of Charging Requests", fontsize=12)
    ax.set_ylabel("Service Rate (%)", fontsize=12)
    ax.set_title("(c) Service Rate vs Demand", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    fig.suptitle("Fleet Requirement vs Demand — Algorithm Comparison\n"
                 "(Heterogeneous Fleet: MCS-Slow + MCS-Fast + UAV)",
                 fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "fig6_fleet_vs_demand.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
