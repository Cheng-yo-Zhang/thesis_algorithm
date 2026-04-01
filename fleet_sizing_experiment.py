"""
Fleet Sizing 實驗：逐一變動單一車型，找出各車型最小需求量
================================================================
三組實驗：
  A. MCS-Slow: 1~10, 固定 Fast=6, UAV=4
  B. MCS-Fast: 1~10, 固定 Slow=6, UAV=4
  C. UAV: 0~6,       固定 Slow=6, Fast=6

每組跑三策略 (EDF / Slack / Regret-2)，觀察 MW service rate。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

from config import Config
from main import run_simulation

OUT_DIR = Path("fleet_sizing_results")
STRATEGIES = ["edf", "slack", "regret2"]
STRATEGY_LABELS = {"edf": "Greedy-EDF", "slack": "Greedy-Slack", "regret2": "Regret-2"}
STRATEGY_COLORS = {"edf": "#9E9E9E", "slack": "#2196F3", "regret2": "#FF5722"}
STRATEGY_MARKERS = {"edf": "^", "slack": "o", "regret2": "s"}


def make_cfg(**overrides) -> Config:
    """建立基準場景 Config，可覆寫參數。"""
    cfg = Config()
    cfg.FIXED_DEMAND_PER_SLOT = 8
    cfg.NUM_MCS_SLOW = 6
    cfg.NUM_MCS_FAST = 6
    cfg.NUM_UAV = 4
    cfg.T_TOTAL = 1440
    cfg.WARMUP_END = 360
    cfg.MEASUREMENT_START = 360
    cfg.MEASUREMENT_END = 1080
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run_sweep(
    param_name: str,
    param_values: List[int],
    label_fn,
) -> Dict[str, Dict[str, list]]:
    """對指定參數做 sweep，回傳各策略的結果。"""
    results = {s: {
        "param_values": [], "labels": [],
        "mw_rate": [], "mw_urgent_rate": [], "mw_normal_rate": [],
        "fleet_used": [], "mw_missed": [],
        "mw_n_slow": [], "mw_n_fast": [], "mw_n_uav": [],
    } for s in STRATEGIES}

    for val in param_values:
        label = label_fn(val)
        print(f"\n  {param_name}={val} ({label})")

        for strat in STRATEGIES:
            cfg = make_cfg(**{param_name: val, "CONSTRUCTION_STRATEGY": strat})
            _, _, _, _, st = run_simulation(cfg, verbose=False)

            r = results[strat]
            r["param_values"].append(val)
            r["labels"].append(label)
            r["mw_rate"].append(st["mw_service_rate"])
            r["mw_urgent_rate"].append(st["mw_urgent_rate"])
            r["mw_normal_rate"].append(st["mw_normal_rate"])
            r["fleet_used"].append(st["mw_fleet_used"])
            r["mw_missed"].append(st["mw_missed"])
            r["mw_n_slow"].append(st["mw_n_slow"])
            r["mw_n_fast"].append(st["mw_n_fast"])
            r["mw_n_uav"].append(st["mw_n_uav"])

            print(f"    {STRATEGY_LABELS[strat]:>15s}: MW={st['mw_service_rate']:.2%}  "
                  f"Urgent={st['mw_urgent_rate']:.2%}  Normal={st['mw_normal_rate']:.2%}  "
                  f"Used=S{st['mw_n_slow']}/F{st['mw_n_fast']}/U{st['mw_n_uav']}  "
                  f"Missed={st['mw_missed']}")

    return results


def plot_sweep(
    results: Dict[str, Dict[str, list]],
    x_label: str,
    title_suffix: str,
    filename_prefix: str,
    out_dir: Path,
):
    """畫 sweep 結果：MW rate + Urgent rate + Fleet used。"""
    x = np.arange(len(results["edf"]["param_values"]))
    x_ticks = results["edf"]["labels"]

    # Fig 1: MW Service Rate
    fig, ax = plt.subplots(figsize=(10, 5))
    for strat in STRATEGIES:
        ax.plot(x, [r * 100 for r in results[strat]["mw_rate"]],
                f"{STRATEGY_MARKERS[strat]}-", color=STRATEGY_COLORS[strat],
                linewidth=2, markersize=8, label=STRATEGY_LABELS[strat])
    ax.axhline(y=95, color="gray", linestyle="--", alpha=0.5, label="Target 95%")
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("MW Service Rate (%)", fontsize=11)
    ax.set_title(f"Fleet Sizing: {title_suffix} — MW Service Rate", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{filename_prefix}_mw_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: Urgent Service Rate
    fig, ax = plt.subplots(figsize=(10, 5))
    for strat in STRATEGIES:
        ax.plot(x, [r * 100 for r in results[strat]["mw_urgent_rate"]],
                f"{STRATEGY_MARKERS[strat]}-", color=STRATEGY_COLORS[strat],
                linewidth=2, markersize=8, label=STRATEGY_LABELS[strat])
    ax.axhline(y=95, color="gray", linestyle="--", alpha=0.5, label="Target 95%")
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Urgent Service Rate (%)", fontsize=11)
    ax.set_title(f"Fleet Sizing: {title_suffix} — Urgent Service Rate", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{filename_prefix}_urgent_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: Actual Vehicles Used (stacked by type)
    fig, ax = plt.subplots(figsize=(10, 5))
    n_strat = len(STRATEGIES)
    width = 0.8 / n_strat
    for i, strat in enumerate(STRATEGIES):
        offset = (i - (n_strat - 1) / 2) * width
        slow = results[strat]["mw_n_slow"]
        fast = results[strat]["mw_n_fast"]
        uav = results[strat]["mw_n_uav"]
        ax.bar(x + offset, slow, width, color="#4CAF50", alpha=0.8,
               label="Slow" if i == 0 else None)
        ax.bar(x + offset, fast, width, bottom=slow, color="#FF9800", alpha=0.8,
               label="Fast" if i == 0 else None)
        ax.bar(x + offset, uav, width,
               bottom=[s + f for s, f in zip(slow, fast)],
               color="#9C27B0", alpha=0.8, label="UAV" if i == 0 else None)
        # 標註策略名
        for j in range(len(x)):
            total = slow[j] + fast[j] + uav[j]
            ax.text(x[j] + offset, total + 0.3,
                    STRATEGY_LABELS[strat].split("-")[-1] if "-" in STRATEGY_LABELS[strat] else "R2",
                    ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Vehicles Used (MW)", fontsize=11)
    ax.set_title(f"Fleet Sizing: {title_suffix} — Vehicles Used by Type", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / f"{filename_prefix}_vehicles_used.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary_table(results: Dict[str, Dict[str, list]], param_name: str):
    """印出摘要表格。"""
    print(f"\n{'='*90}")
    print(f"  Summary: {param_name}")
    print(f"{'='*90}")
    header = f"{'Config':<10} {'Strategy':>15} {'MW%':>8} {'Urg%':>8} {'Nor%':>8} {'Miss':>6} {'S':>3} {'F':>3} {'U':>3} {'Tot':>4}"
    print(header)
    print("-" * len(header))

    for i in range(len(results["edf"]["param_values"])):
        for strat in STRATEGIES:
            r = results[strat]
            print(f"{r['labels'][i]:<10} {STRATEGY_LABELS[strat]:>15} "
                  f"{r['mw_rate'][i]:>7.2%} {r['mw_urgent_rate'][i]:>7.2%} "
                  f"{r['mw_normal_rate'][i]:>7.2%} {r['mw_missed'][i]:>5d}  "
                  f"{r['mw_n_slow'][i]:>2d}  {r['mw_n_fast'][i]:>2d}  {r['mw_n_uav'][i]:>2d}  "
                  f"{r['fleet_used'][i]:>3d}")
        print()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # === Experiment A: MCS-Slow sweep ===
    print("\n" + "=" * 70)
    print("  Experiment A: MCS-Slow sweep (Fast=6, UAV=4)")
    print("=" * 70)
    results_a = run_sweep(
        "NUM_MCS_SLOW", list(range(1, 11)),
        label_fn=lambda v: f"S{v}"
    )
    plot_sweep(results_a, "MCS-Slow Count", "MCS-Slow Sweep (Fast=6, UAV=4)",
               "A_slow_sweep", OUT_DIR)
    print_summary_table(results_a, "MCS-Slow")

    # === Experiment B: MCS-Fast sweep ===
    print("\n" + "=" * 70)
    print("  Experiment B: MCS-Fast sweep (Slow=6, UAV=4)")
    print("=" * 70)
    results_b = run_sweep(
        "NUM_MCS_FAST", list(range(1, 11)),
        label_fn=lambda v: f"F{v}"
    )
    plot_sweep(results_b, "MCS-Fast Count", "MCS-Fast Sweep (Slow=6, UAV=4)",
               "B_fast_sweep", OUT_DIR)
    print_summary_table(results_b, "MCS-Fast")

    # === Experiment C: UAV sweep ===
    print("\n" + "=" * 70)
    print("  Experiment C: UAV sweep (Slow=6, Fast=6)")
    print("=" * 70)
    results_c = run_sweep(
        "NUM_UAV", list(range(0, 7)),
        label_fn=lambda v: f"U{v}"
    )
    plot_sweep(results_c, "UAV Count", "UAV Sweep (Slow=6, Fast=6)",
               "C_uav_sweep", OUT_DIR)
    print_summary_table(results_c, "UAV")

    print(f"\n{'='*70}")
    print(f"  All outputs saved to: {OUT_DIR.resolve()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
