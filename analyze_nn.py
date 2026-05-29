"""Diagnose NN cumulative-time jitter / plateau pattern in route_events.csv."""
import csv
from collections import defaultdict

PATH = r"C:\Users\Louis\thesis_algorithm\results\route_comparison\route_events.csv"

with open(PATH, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))


def f(x):
    return float(x)


def show(alg):
    sub = sorted([r for r in rows if r["algorithm"] == alg],
                 key=lambda r: int(r["serve_rank"]))
    print(f"\n========= {alg} =========")
    print(f"{'rank':>4} {'veh':>3} {'vtype':>9} {'node':>4} {'ntype':>6} "
          f"{'arr':>7} {'dep':>7} {'dt':>7} {'wait_u':>6} {'wait_m':>6} {'hop':>6}")
    prev = 0.0
    for r in sub:
        dep = f(r["departure_time"])
        dt = dep - prev
        prev = dep
        print(f"{r['serve_rank']:>4} {r['vehicle_id']:>3} {r['vehicle_type']:>9} "
              f"{r['node_id']:>4} {r['node_type'][:6]:>6} "
              f"{f(r['arrival_time']):>7.2f} {dep:>7.2f} {dt:>7.2f} "
              f"{f(r['user_waiting_time']):>6.2f} {f(r['mcs_waiting_time']):>6.2f} "
              f"{f(r['hop_distance']):>6.2f}")


def per_vehicle(alg):
    sub = [r for r in rows if r["algorithm"] == alg]
    by_v = defaultdict(list)
    for r in sub:
        by_v[(r["vehicle_type"], r["vehicle_id"])].append(r)
    print(f"\n----- {alg} per-vehicle timeline -----")
    for key, items in sorted(by_v.items()):
        items.sort(key=lambda r: int(r["route_position"]))
        ranks = [int(x["serve_rank"]) for x in items]
        deps = [f(x["departure_time"]) for x in items]
        print(f"  vehicle {key}: served {len(items)} customers, "
              f"ranks {ranks[:6]}...{ranks[-6:]}, "
              f"first dep {deps[0]:.1f}, last dep {deps[-1]:.1f}, span {deps[-1]-deps[0]:.1f}")


def vehicle_active_window(alg):
    """For each vehicle, find when it stops serving."""
    sub = [r for r in rows if r["algorithm"] == alg]
    by_v = defaultdict(list)
    for r in sub:
        by_v[(r["vehicle_type"], r["vehicle_id"])].append(r)
    print(f"\n----- {alg} vehicle active spans -----")
    for key, items in sorted(by_v.items()):
        items.sort(key=lambda r: f(r["departure_time"]))
        last_rank = max(int(x["serve_rank"]) for x in items)
        last_dep = max(f(x["departure_time"]) for x in items)
        first_dep = min(f(x["departure_time"]) for x in items)
        print(f"  {key}: count={len(items)}, "
              f"first_dep={first_dep:.1f}, last_dep={last_dep:.1f}, last_rank={last_rank}")


for alg in ("NN", "TCGI", "TCGI-ALNS"):
    show(alg)
    per_vehicle(alg)
    vehicle_active_window(alg)
