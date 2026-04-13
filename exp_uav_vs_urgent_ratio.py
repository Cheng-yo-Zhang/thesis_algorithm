"""
Urgent Miss Rate vs Urgent Ratio — UAV value demonstration.

Fixed:
    n_requests = 20  (single-batch static dispatch)
    fleet = (MCS_SLOW=4, MCS_FAST=6)
    construction = regret2 + ALNS (1000 iter)
    seed = 42 (single replication)

Sweep:
    URGENT_RATIO  ∈ [0.1, 0.2, ..., 0.9]
    NUM_UAV       ∈ [0, 4]     → two lines on the plot

Output:
    results_uav_vs_urgent_ratio/raw.csv
    results_uav_vs_urgent_ratio/urgent_miss_rate.png
    results_uav_vs_urgent_ratio/urgent_miss_rate.pdf
"""

import csv
import time
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from problem import ChargingSchedulingProblem
from simulation import initialize_fleet
from exp_fleet_vs_demand import generate_static_requests, solve_once


# === Experiment Parameters ===
URGENT_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_UAV_LIST = [0, 1]
SEED = 42
N_REQUESTS = 20
NUM_MCS_SLOW = 3
NUM_MCS_FAST = 2
ALNS_ITER = 1000

OUTPUT_DIR = Path("results_uav_vs_urgent_ratio")
STRATEGY = {"construction": "regret2", "alns_iter": ALNS_ITER}


def run_one(rho: float, num_uav: int, seed: int) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    cfg = Config(
        RANDOM_SEED=seed,
        URGENT_RATIO=rho,
        NUM_MCS_SLOW=NUM_MCS_SLOW,
        NUM_MCS_FAST=NUM_MCS_FAST,
        NUM_UAV=num_uav,
        CONSTRUCTION_STRATEGY="regret2",
        ALNS_MAX_ITERATIONS=ALNS_ITER,
    )
    problem = ChargingSchedulingProblem(cfg)
    fleet = initialize_fleet(cfg, problem.depot)
    requests = generate_static_requests(problem, N_REQUESTS, cfg)
    problem.setup_nodes(requests)

    t0 = time.time()
    solution = solve_once(requests, fleet, problem, cfg, STRATEGY)
    elapsed = time.time() - t0

    # Force populate cost + waiting-time fields
    solution.calculate_total_cost(total_customers=N_REQUESTS)

    urgent_total = sum(1 for r in requests if r.node_type == 'urgent')
    normal_total = sum(1 for r in requests if r.node_type == 'normal')
    urgent_missed = sum(1 for n in solution.unassigned_nodes if n.node_type == 'urgent')
    normal_missed = sum(1 for n in solution.unassigned_nodes if n.node_type == 'normal')

    urgent_miss_rate = urgent_missed / urgent_total if urgent_total > 0 else 0.0
    normal_miss_rate = normal_missed / normal_total if normal_total > 0 else 0.0

    # Task split: urgent served by vehicle type
    urgent_by_slow = 0
    urgent_by_fast = 0
    urgent_by_uav = 0
    urgent_wait_sum = 0.0
    normal_wait_sum = 0.0
    for route in solution.get_all_routes():
        for i, node in enumerate(route.nodes):
            wt = route.user_waiting_times[i] if i < len(route.user_waiting_times) else 0.0
            if node.node_type == 'urgent':
                if route.vehicle_type == 'mcs_slow':
                    urgent_by_slow += 1
                elif route.vehicle_type == 'mcs_fast':
                    urgent_by_fast += 1
                elif route.vehicle_type == 'uav':
                    urgent_by_uav += 1
                urgent_wait_sum += wt
            else:
                normal_wait_sum += wt

    # Cost decomposition (mirrors models.py:calculate_total_cost)
    miss_penalty = cfg.ALPHA_URGENT * urgent_missed + cfg.ALPHA_NORMAL * normal_missed
    waiting_cost = (cfg.BETA_WAITING_URGENT * urgent_wait_sum
                    + cfg.BETA_WAITING_NORMAL * normal_wait_sum)
    travel_cost = cfg.GAMMA_DISTANCE * solution.total_distance

    return {
        "rho": rho,
        "num_uav": num_uav,
        "seed": seed,
        "n_requests": N_REQUESTS,
        "urgent_total": urgent_total,
        "normal_total": normal_total,
        "urgent_missed": urgent_missed,
        "normal_missed": normal_missed,
        "urgent_miss_rate": urgent_miss_rate,
        "normal_miss_rate": normal_miss_rate,
        "urgent_by_slow": urgent_by_slow,
        "urgent_by_fast": urgent_by_fast,
        "urgent_by_uav": urgent_by_uav,
        "avg_wait_urgent": solution.avg_waiting_time_urgent,
        "avg_wait_normal": solution.avg_waiting_time_normal,
        "total_distance": solution.total_distance,
        "total_cost": solution.total_cost,
        "miss_penalty": miss_penalty,
        "waiting_cost": waiting_cost,
        "travel_cost": travel_cost,
        "elapsed_sec": elapsed,
    }


def run_sweep() -> list:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    total_runs = len(URGENT_RATIOS) * len(NUM_UAV_LIST)
    run_idx = 0
    t_start = time.time()

    print("=" * 72)
    print("Urgent Miss Rate vs Urgent Ratio")
    print(f"  n_requests={N_REQUESTS}  fleet=(slow={NUM_MCS_SLOW}, fast={NUM_MCS_FAST})")
    print(f"  rho grid  = {URGENT_RATIOS}")
    print(f"  num_uav   = {NUM_UAV_LIST}")
    print(f"  seed      = {SEED}")
    print(f"  ALNS iter = {ALNS_ITER}")
    print(f"  total runs= {total_runs}")
    print("=" * 72)

    for num_uav in NUM_UAV_LIST:
        for rho in URGENT_RATIOS:
            run_idx += 1
            row = run_one(rho, num_uav, SEED)
            rows.append(row)
            print(
                f"  [{run_idx:2d}/{total_runs}] rho={rho:.1f} uav={num_uav} | "
                f"urgent={row['urgent_total']:2d} miss={row['urgent_missed']:2d} "
                f"rate={row['urgent_miss_rate']:.3f} | "
                f"uav_urg={row['urgent_by_uav']:2d} | "
                f"{row['elapsed_sec']:.2f}s"
            )

    total_elapsed = time.time() - t_start
    print("=" * 72)
    print(f"Sweep done in {total_elapsed:.1f}s")

    csv_path = OUTPUT_DIR / "raw.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {csv_path}")

    return rows


LINE_STYLES = {
    0: dict(label="Baseline (No UAV)", linestyle="--", marker="o",
            color="#7f8c8d"),
    1: dict(label="With UAV (K=1)", linestyle="-", marker="s",
            color="#e74c3c"),
    2: dict(label="With UAV (K=2)", linestyle="-", marker="s",
            color="#e74c3c"),
    4: dict(label="With UAV (K=4)", linestyle="-", marker="s",
            color="#e74c3c"),
}


def _group_by_uav(rows: list) -> dict:
    by_uav = {k: [] for k in NUM_UAV_LIST}
    for r in rows:
        by_uav[r["num_uav"]].append(r)
    for k in by_uav:
        by_uav[k].sort(key=lambda r: r["rho"])
    return by_uav


def _save(fig, name: str) -> None:
    png_path = OUTPUT_DIR / f"{name}.png"
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {png_path.name}")


def _line_plot(by_uav: dict, key: str, ylabel: str, title: str, name: str,
               ylim_bottom=0, ylim_top=None) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in NUM_UAV_LIST:
        xs = [r["rho"] for r in by_uav[k]]
        ys = [r[key] for r in by_uav[k]]
        style = LINE_STYLES.get(
            k, dict(label=f"UAV={k}", linestyle="-", marker="^", color="#e74c3c")
        )
        ax.plot(xs, ys, linewidth=1.8, markersize=6, **style)

    ax.set_xlabel(r"Urgent Ratio $\rho$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
    ax.set_xticks(URGENT_RATIOS)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    _save(fig, name)


def plot_urgent_miss_rate(by_uav: dict) -> None:
    _line_plot(by_uav, "urgent_miss_rate", "Urgent Miss Rate",
               "UAV Impact on Urgent Miss Rate", "urgent_miss_rate")


def plot_urgent_waiting_time(by_uav: dict) -> None:
    _line_plot(by_uav, "avg_wait_urgent", "Avg Urgent Waiting Time (min)",
               "UAV Impact on Urgent Waiting Time", "urgent_waiting_time")


def plot_normal_miss_rate(by_uav: dict) -> None:
    _line_plot(by_uav, "normal_miss_rate", "Normal Miss Rate",
               "UAV Side Effect on Normal Miss Rate", "normal_miss_rate")


def plot_total_cost(by_uav: dict) -> None:
    _line_plot(by_uav, "total_cost", "Total Cost Z",
               "UAV Impact on Total Cost", "total_cost")


def plot_task_split_stacked(by_uav: dict) -> None:
    """Stacked area: urgent served by vehicle type. Two subplots — baseline vs UAV."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    layer_colors = {
        "mcs_slow": "#2ecc71",
        "mcs_fast": "#e67e22",
        "uav": "#9b59b6",
    }
    titles = {0: "Baseline (No UAV)", NUM_UAV_LIST[1]: f"With UAV (K={NUM_UAV_LIST[1]})"}

    for ax, k in zip(axes, NUM_UAV_LIST):
        xs = [r["rho"] for r in by_uav[k]]
        slow = np.array([r["urgent_by_slow"] for r in by_uav[k]])
        fast = np.array([r["urgent_by_fast"] for r in by_uav[k]])
        uav = np.array([r["urgent_by_uav"] for r in by_uav[k]])
        missed = np.array([r["urgent_missed"] for r in by_uav[k]])

        ax.stackplot(
            xs, slow, fast, uav, missed,
            labels=["MCS-Slow", "MCS-Fast", "UAV", "Missed"],
            colors=[layer_colors["mcs_slow"], layer_colors["mcs_fast"],
                    layer_colors["uav"], "#cccccc"],
            alpha=0.85,
        )
        ax.set_xlabel(r"Urgent Ratio $\rho$", fontsize=12)
        ax.set_title(titles.get(k, f"K={k}"), fontsize=13)
        ax.set_xlim(0.1, 0.9)
        ax.set_xticks(URGENT_RATIOS)
        ax.grid(True, linestyle=":", alpha=0.5)
        if k == 0:
            ax.set_ylabel("Urgent Requests (count)", fontsize=12)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.suptitle("Urgent Task Allocation by Vehicle Type", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "task_split_stacked")


def plot_all(rows: list) -> None:
    by_uav = _group_by_uav(rows)
    print("Plotting figures:")
    plot_urgent_miss_rate(by_uav)
    plot_urgent_waiting_time(by_uav)
    plot_normal_miss_rate(by_uav)
    plot_total_cost(by_uav)
    plot_task_split_stacked(by_uav)


if __name__ == "__main__":
    rows = run_sweep()
    plot_all(rows)
