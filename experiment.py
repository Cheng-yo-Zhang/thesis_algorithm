"""
Sensitivity Analysis: ρ (URGENT_RATIO) vs Fleet Requirement
=============================================================
給定足夠大的車隊 pool，讓 dispatch 自行決定使用多少車輛。
對不同 ρ 值，統計測量窗口 (hour 12-36) 內實際使用的車輛數量。
"""

import numpy as np
import csv
import time
from math import sqrt

from config import Config
from main import run_simulation


def run_sensitivity_analysis(
    rho_values: list = None,
    num_replications: int = 10,
    base_seed: int = 42,
    output_csv: str = "sensitivity_results.csv",
):
    """
    對每個 ρ 值跑 num_replications 次模擬，
    統計測量窗口內的服務率和實際車輛使用量。
    """
    if rho_values is None:
        rho_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"{'='*80}")
    print(f"Sensitivity Analysis: ρ (URGENT_RATIO) vs Fleet Requirement")
    print(f"{'='*80}")
    print(f"  ρ values: {rho_values}")
    print(f"  Replications: {num_replications}")
    print(f"  Fleet pool: sufficient (20 each type)")
    print()

    t_start = time.time()
    results = []

    for rho in rho_values:
        rep_data = {
            "rates": [], "urgent_rates": [], "normal_rates": [],
            "n_slow": [], "n_fast": [], "n_uav": [], "fleet_total": [],
        }

        for rep in range(num_replications):
            cfg = Config(
                RANDOM_SEED=base_seed + rep,
                URGENT_RATIO=rho,
            )
            _, _, _, _, stats = run_simulation(cfg, verbose=False)

            rep_data["rates"].append(stats["mw_service_rate"])
            rep_data["urgent_rates"].append(stats["mw_urgent_rate"])
            rep_data["normal_rates"].append(stats["mw_normal_rate"])
            rep_data["n_slow"].append(stats["mw_n_slow"])
            rep_data["n_fast"].append(stats["mw_n_fast"])
            rep_data["n_uav"].append(stats["mw_n_uav"])
            rep_data["fleet_total"].append(stats["mw_fleet_used"])

        def _ci(arr):
            m = np.mean(arr)
            s = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
            h = 1.96 * s / sqrt(len(arr)) if len(arr) > 1 else 0.0
            return m, s, m - h, m + h

        rate_m, rate_s, rate_lo, rate_hi = _ci(rep_data["rates"])
        urg_m, _, _, _ = _ci(rep_data["urgent_rates"])
        nor_m, _, _, _ = _ci(rep_data["normal_rates"])
        slow_m, slow_s, _, _ = _ci(rep_data["n_slow"])
        fast_m, fast_s, _, _ = _ci(rep_data["n_fast"])
        uav_m, uav_s, _, _ = _ci(rep_data["n_uav"])
        fleet_m, fleet_s, _, _ = _ci(rep_data["fleet_total"])

        row = {
            "rho": rho,
            "mean_rate": rate_m,
            "std_rate": rate_s,
            "ci_lower": rate_lo,
            "ci_upper": rate_hi,
            "mean_urgent_rate": urg_m,
            "mean_normal_rate": nor_m,
            "mean_n_slow": slow_m,
            "std_n_slow": slow_s,
            "mean_n_fast": fast_m,
            "std_n_fast": fast_s,
            "mean_n_uav": uav_m,
            "std_n_uav": uav_s,
            "mean_fleet_total": fleet_m,
            "std_fleet_total": fleet_s,
        }
        results.append(row)

        elapsed = time.time() - t_start
        print(f"  ρ={rho:.1f} | rate={rate_m:.3f}±{rate_s:.3f} "
              f"| urgent={urg_m:.3f} normal={nor_m:.3f} "
              f"| fleet: S={slow_m:.1f} F={fast_m:.1f} U={uav_m:.1f} "
              f"total={fleet_m:.1f}±{fleet_s:.1f} "
              f"[{elapsed:.0f}s]")

    # CSV 輸出
    fieldnames = list(results[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_csv}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"Summary: ρ vs Fleet Requirement")
    print(f"{'='*80}")
    print(f"  {'ρ':>4} {'Rate':>8} {'Urgent':>8} {'Normal':>8} "
          f"{'Slow':>6} {'Fast':>6} {'UAV':>6} {'Total':>7}")
    print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for r in results:
        print(f"  {r['rho']:4.1f} {r['mean_rate']:8.3f} "
              f"{r['mean_urgent_rate']:8.3f} {r['mean_normal_rate']:8.3f} "
              f"{r['mean_n_slow']:6.1f} {r['mean_n_fast']:6.1f} {r['mean_n_uav']:6.1f} "
              f"{r['mean_fleet_total']:7.1f}")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time/60:.1f} min")

    return results


def run_strategy_comparison(
    rho_values: list = None,
    strategies: list = None,
    num_replications: int = 10,
    base_seed: int = 42,
    output_csv: str = "strategy_comparison.csv",
):
    """
    對比不同 CONSTRUCTION_STRATEGY 在各 ρ 下的表現。
    同一組 (rho, seed) 分別跑不同策略，確保公平比較。
    """
    if rho_values is None:
        rho_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    if strategies is None:
        strategies = ["nearest", "deadline", "alns"]

    print(f"{'='*80}")
    print(f"Strategy Comparison: {strategies}")
    print(f"{'='*80}")
    print(f"  ρ values: {rho_values}")
    print(f"  Replications: {num_replications}")
    print()

    t_start = time.time()
    results = []

    for rho in rho_values:
        for strategy in strategies:
            rep_data = {
                "rates": [], "urgent_rates": [], "normal_rates": [],
                "n_slow": [], "n_fast": [], "n_uav": [], "fleet_total": [],
            }

            for rep in range(num_replications):
                cfg = Config(
                    RANDOM_SEED=base_seed + rep,
                    URGENT_RATIO=rho,
                    CONSTRUCTION_STRATEGY=strategy,
                )
                _, _, _, _, stats = run_simulation(cfg, verbose=False)

                rep_data["rates"].append(stats["mw_service_rate"])
                rep_data["urgent_rates"].append(stats["mw_urgent_rate"])
                rep_data["normal_rates"].append(stats["mw_normal_rate"])
                rep_data["n_slow"].append(stats["mw_n_slow"])
                rep_data["n_fast"].append(stats["mw_n_fast"])
                rep_data["n_uav"].append(stats["mw_n_uav"])
                rep_data["fleet_total"].append(stats["mw_fleet_used"])

            def _ci(arr):
                m = np.mean(arr)
                s = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
                h = 1.96 * s / sqrt(len(arr)) if len(arr) > 1 else 0.0
                return m, s, m - h, m + h

            rate_m, rate_s, rate_lo, rate_hi = _ci(rep_data["rates"])
            urg_m, _, _, _ = _ci(rep_data["urgent_rates"])
            nor_m, _, _, _ = _ci(rep_data["normal_rates"])
            slow_m, slow_s, _, _ = _ci(rep_data["n_slow"])
            fast_m, fast_s, _, _ = _ci(rep_data["n_fast"])
            uav_m, uav_s, _, _ = _ci(rep_data["n_uav"])
            fleet_m, fleet_s, _, _ = _ci(rep_data["fleet_total"])

            row = {
                "strategy": strategy,
                "rho": rho,
                "mean_rate": rate_m,
                "std_rate": rate_s,
                "ci_lower": rate_lo,
                "ci_upper": rate_hi,
                "mean_urgent_rate": urg_m,
                "mean_normal_rate": nor_m,
                "mean_fleet_total": fleet_m,
                "std_fleet_total": fleet_s,
            }
            results.append(row)

            elapsed = time.time() - t_start
            print(f"  [{strategy:>8}] ρ={rho:.1f} | rate={rate_m:.3f}±{rate_s:.3f} "
                  f"| urgent={urg_m:.3f} normal={nor_m:.3f} "
                  f"| fleet={fleet_m:.1f} [{elapsed:.0f}s]")

    # CSV 輸出
    fieldnames = list(results[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_csv}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"Strategy Comparison Summary")
    print(f"{'='*80}")
    print(f"  {'Strategy':>8} {'ρ':>4} {'Rate':>8} {'Urgent':>8} {'Normal':>8} {'Fleet':>7}")
    print(f"  {'-'*8} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
    for r in results:
        print(f"  {r['strategy']:>8} {r['rho']:4.1f} {r['mean_rate']:8.3f} "
              f"{r['mean_urgent_rate']:8.3f} {r['mean_normal_rate']:8.3f} "
              f"{r['mean_fleet_total']:7.1f}")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time/60:.1f} min")

    # 產生比較圖表
    from pathlib import Path
    from visualization import plot_strategy_comparison
    plot_strategy_comparison(output_csv, Path("visualization_output"))

    return results


if __name__ == "__main__":
    results = run_strategy_comparison(
        rho_values=[0.1, 0.3, 0.5, 0.7, 0.9],
        strategies=["nearest", "deadline", "alns"],
        num_replications=1,
    )
