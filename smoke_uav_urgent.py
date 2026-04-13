"""Smoke test: single (rho, num_uav, seed) run to verify pipeline + ALNS runtime."""
import time
import numpy as np
import random

from config import Config
from problem import ChargingSchedulingProblem
from simulation import initialize_fleet
from exp_fleet_vs_demand import generate_static_requests, solve_once


def run_single(rho, num_uav, seed, n_req=20, alns_iter=1000,
               num_slow=4, num_fast=6):
    np.random.seed(seed)
    random.seed(seed)

    cfg = Config(
        RANDOM_SEED=seed,
        URGENT_RATIO=rho,
        NUM_MCS_SLOW=num_slow,
        NUM_MCS_FAST=num_fast,
        NUM_UAV=num_uav,
        CONSTRUCTION_STRATEGY="regret2",
        ALNS_MAX_ITERATIONS=alns_iter,
    )
    problem = ChargingSchedulingProblem(cfg)
    fleet = initialize_fleet(cfg, problem.depot)

    requests = generate_static_requests(problem, n_req, cfg)
    problem.setup_nodes(requests)

    t0 = time.time()
    solution = solve_once(
        requests, fleet, problem, cfg,
        strategy={"construction": "regret2", "alns_iter": alns_iter},
    )
    elapsed = time.time() - t0

    urgent_total = sum(1 for r in requests if r.node_type == 'urgent')
    normal_total = sum(1 for r in requests if r.node_type == 'normal')
    urgent_missed = sum(1 for n in solution.unassigned_nodes if n.node_type == 'urgent')
    normal_missed = sum(1 for n in solution.unassigned_nodes if n.node_type == 'normal')
    urgent_served = urgent_total - urgent_missed

    urgent_miss_rate = urgent_missed / urgent_total if urgent_total > 0 else 0.0

    n_uav_used = sum(1 for r in solution.uav_routes if len(r.nodes) > 0)
    urgent_via_uav = sum(
        1 for r in solution.uav_routes
        for n in r.nodes if n.node_type == 'urgent'
    )

    print(f"  rho={rho} num_uav={num_uav} seed={seed}")
    print(f"    n_req={n_req}  urgent={urgent_total}  normal={normal_total}")
    print(f"    urgent_missed={urgent_missed}  normal_missed={normal_missed}")
    print(f"    urgent_miss_rate={urgent_miss_rate:.3f}")
    print(f"    uav_routes_used={n_uav_used}  urgent_served_by_uav={urgent_via_uav}")
    print(f"    elapsed={elapsed:.2f}s")
    return elapsed


if __name__ == "__main__":
    print("=" * 60)
    print("Smoke test: n=20, ALNS 1000 iter, fleet (slow=4, fast=6)")
    print("=" * 60)
    print("\n[1] Baseline (no UAV) at rho=0.5")
    t1 = run_single(rho=0.5, num_uav=0, seed=42)
    print("\n[2] With UAV (K=4) at rho=0.5")
    t2 = run_single(rho=0.5, num_uav=4, seed=42)
    print("\n[3] Baseline at rho=0.9 (stress test)")
    t3 = run_single(rho=0.9, num_uav=0, seed=42)

    avg = (t1 + t2 + t3) / 3
    print("\n" + "=" * 60)
    print(f"Avg elapsed: {avg:.2f}s")
    print(f"Projected main sweep (18 runs): {avg * 18:.1f}s = {avg * 18 / 60:.1f} min")
    print("=" * 60)
