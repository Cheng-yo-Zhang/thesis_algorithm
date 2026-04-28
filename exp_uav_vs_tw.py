"""
Urgent Miss Rate vs Urgent Time Window — problem-hardness sensitivity.

Fixed:
    n_requests    = 20   (single-batch static dispatch)
    URGENT_RATIO  = 0.5
    fleet         = (MCS_SLOW=3, MCS_FAST=2)
    construction  = regret2 + ALNS (1000 iter)
    seed          = 42

Sweep:
    URGENT_TW ∈ [15, 20, 30, 45, 60, 90, 120, 180]   (min, deterministic)
    NUM_UAV   ∈ [0, 1]                               → two lines

Each TW value sets BOTH URGENT_TW_MIN and URGENT_TW_MAX to the same value,
so every urgent request gets exactly that time window (no sampling noise).

Output:
    results/uav_vs_tw/urgent_miss_rate.png
"""

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
TW_VALUES = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
NUM_UAV_LIST = [0, 1, 2, 3]
SEED = 42
N_REQUESTS = 20
NUM_MCS_SLOW = 3
NUM_MCS_FAST = 2
URGENT_RATIO = 0.5
ALNS_ITER = 5000

OUTPUT_DIR = Path("results") / "uav_vs_tw"
STRATEGY = {"construction": "regret2", "alns_iter": ALNS_ITER}


def run_one(tw: float, num_uav: int, seed: int) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    cfg = Config(
        RANDOM_SEED=seed,
        URGENT_RATIO=URGENT_RATIO,
        URGENT_TW_MIN=float(tw),
        URGENT_TW_MAX=float(tw),
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

    solution.calculate_total_cost(total_customers=N_REQUESTS)

    urgent_total = sum(1 for r in requests if r.node_type == 'urgent')
    normal_total = sum(1 for r in requests if r.node_type == 'normal')
    urgent_missed = sum(1 for n in solution.unassigned_nodes if n.node_type == 'urgent')
    normal_missed = sum(1 for n in solution.unassigned_nodes if n.node_type == 'normal')

    urgent_miss_rate = urgent_missed / urgent_total if urgent_total > 0 else 0.0

    urgent_by_uav = sum(
        1 for r in solution.uav_routes
        for n in r.nodes if n.node_type == 'urgent'
    )

    return {
        "tw": tw,
        "num_uav": num_uav,
        "seed": seed,
        "n_requests": N_REQUESTS,
        "urgent_ratio": URGENT_RATIO,
        "urgent_total": urgent_total,
        "normal_total": normal_total,
        "urgent_missed": urgent_missed,
        "normal_missed": normal_missed,
        "urgent_miss_rate": urgent_miss_rate,
        "urgent_by_uav": urgent_by_uav,
        "total_cost": solution.total_cost,
        "elapsed_sec": elapsed,
    }


def run_sweep() -> list:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    total_runs = len(TW_VALUES) * len(NUM_UAV_LIST)
    run_idx = 0
    t_start = time.time()

    print("=" * 72)
    print("Urgent Miss Rate vs Urgent Time Window")
    print(f"  n_requests={N_REQUESTS}  fleet=(slow={NUM_MCS_SLOW}, fast={NUM_MCS_FAST})")
    print(f"  urgent_ratio={URGENT_RATIO}")
    print(f"  TW grid   = {TW_VALUES}")
    print(f"  num_uav   = {NUM_UAV_LIST}")
    print(f"  seed      = {SEED}")
    print(f"  ALNS iter = {ALNS_ITER}")
    print(f"  total runs= {total_runs}")
    print("=" * 72)

    for num_uav in NUM_UAV_LIST:
        for tw in TW_VALUES:
            run_idx += 1
            row = run_one(tw, num_uav, SEED)
            rows.append(row)
            print(
                f"  [{run_idx:2d}/{total_runs}] tw={tw:3d} uav={num_uav} | "
                f"urgent={row['urgent_total']:2d} miss={row['urgent_missed']:2d} "
                f"rate={row['urgent_miss_rate']:.3f} | "
                f"uav_urg={row['urgent_by_uav']:2d} | "
                f"{row['elapsed_sec']:.2f}s"
            )

    total_elapsed = time.time() - t_start
    print("=" * 72)
    print(f"Sweep done in {total_elapsed:.1f}s")

    return rows


def plot(rows: list) -> None:
    by_uav = {k: [] for k in NUM_UAV_LIST}
    for r in rows:
        by_uav[r["num_uav"]].append(r)
    for k in by_uav:
        by_uav[k].sort(key=lambda r: r["tw"])

    styles = {
        0: dict(label="Baseline (No UAV)", linestyle="--", marker="o",
                color="#7f8c8d"),
        1: dict(label="With UAV (K=1)", linestyle="-", marker="s",
                color="#f39c12"),
        2: dict(label="With UAV (K=2)", linestyle="-", marker="^",
                color="#e74c3c"),
        3: dict(label="With UAV (K=3)", linestyle="-", marker="D",
                color="#8e44ad"),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    for k in NUM_UAV_LIST:
        xs = [r["tw"] for r in by_uav[k]]
        ys = [r["urgent_miss_rate"] for r in by_uav[k]]
        style = styles.get(k, dict(label=f"UAV={k}", linestyle="-", marker="^",
                                    color="#e74c3c"))
        ax.plot(xs, ys, linewidth=1.8, markersize=6, **style)

    ax.set_xlabel("Urgent Time Window (min)", fontsize=12)
    ax.set_ylabel("Urgent Miss Rate", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.set_xticks(TW_VALUES)
    ax.grid(True, linestyle=":", alpha=0.6, which="major")
    ax.legend(loc="upper right", frameon=True)
    ax.set_title("UAV Value vs Problem Hardness (Time Window)", fontsize=13)

    fig.tight_layout()
    png_path = OUTPUT_DIR / "urgent_miss_rate.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {png_path}")


if __name__ == "__main__":
    rows = run_sweep()
    plot(rows)
