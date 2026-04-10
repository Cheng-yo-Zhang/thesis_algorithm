"""
Recover plots from the experiment log (one-shot rescue).
解析 b1xo3ermc.output 取得每個 (demand, seed) 的 coverage 與 LCT,
然後產生 CSV 與兩張 PNG, 不需要重跑 30 分鐘的實驗。
"""
import re
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_PATH = Path(
    "C:/Users/Louis/AppData/Local/Temp/claude/"
    "C--Users-Louis-thesis-algorithm/479dd244-2141-4216-acf9-3806f3bc6777/"
    "tasks/b1xo3ermc.output"
)
OUTPUT_DIR = Path("results_nn_regret2_alns")
OUTPUT_DIR.mkdir(exist_ok=True)

DEMAND_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80]
NUM_MCS_SLOW, NUM_MCS_FAST, NUM_UAV = 3, 2, 1

# 解析格式:
#   D=10  seed=0  NN=100.0%(LCT= 198.4)  R2=100.0%(LCT= 132.0)  R2+ALNS=100.0%(LCT=  85.6)  (1.5s)
LINE_RE = re.compile(
    r"D=\s*(\d+)\s+seed=(\d+)\s+"
    r"NN=\s*([\d.]+)%\(LCT=\s*([\d.]+)\)\s+"
    r"R2=\s*([\d.]+)%\(LCT=\s*([\d.]+)\)\s+"
    r"R2\+ALNS=\s*([\d.]+)%\(LCT=\s*([\d.]+)\)"
)

methods = ["nn", "regret2", "regret2_alns"]
all_cov = {m: {d: [] for d in DEMAND_LEVELS} for m in methods}
all_lct = {m: {d: [] for d in DEMAND_LEVELS} for m in methods}
raw_rows = []

with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

for m in LINE_RE.finditer(text):
    d = int(m.group(1))
    seed = int(m.group(2))
    nn_cov, nn_lct = float(m.group(3)) / 100.0, float(m.group(4))
    r2_cov, r2_lct = float(m.group(5)) / 100.0, float(m.group(6))
    al_cov, al_lct = float(m.group(7)) / 100.0, float(m.group(8))

    all_cov["nn"][d].append(nn_cov)
    all_cov["regret2"][d].append(r2_cov)
    all_cov["regret2_alns"][d].append(al_cov)
    all_lct["nn"][d].append(nn_lct)
    all_lct["regret2"][d].append(r2_lct)
    all_lct["regret2_alns"][d].append(al_lct)

    raw_rows.append({
        "demand": d, "seed": seed,
        "nn_coverage": nn_cov, "nn_last_completion": nn_lct,
        "regret2_coverage": r2_cov, "regret2_last_completion": r2_lct,
        "regret2_alns_coverage": al_cov, "regret2_alns_last_completion": al_lct,
    })

print(f"Parsed {len(raw_rows)} (demand, seed) records.")
for d in DEMAND_LEVELS:
    n = len(all_cov["nn"][d])
    print(f"  D={d:>2}: {n} seeds")

# ── CSV ──────────────────────────────────────────────────────────
csv_path = OUTPUT_DIR / "nn_regret2_alns.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
    writer.writeheader()
    writer.writerows(raw_rows)
print(f"CSV saved: {csv_path}")

# ── Plot helpers ─────────────────────────────────────────────────
STYLES = [
    ("nn",           "Nearest-Neighbor", "#7f8c8d", "s"),
    ("regret2",      "Regret-2",         "#3498db", "^"),
    ("regret2_alns", "Regret-2 + ALNS",  "#e74c3c", "o"),
]

def _plot(metric_dict, ylabel, title, filename, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, label, color, marker in STYLES:
        means = [np.mean(metric_dict[method][d]) for d in DEMAND_LEVELS]
        stds  = [np.std(metric_dict[method][d])  for d in DEMAND_LEVELS]
        ax.plot(DEMAND_LEVELS, means, label=label, color=color,
                marker=marker, linewidth=2, markersize=8)
        ax.fill_between(DEMAND_LEVELS,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.15)
    ax.set_xlabel("Number of Requests", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(DEMAND_LEVELS)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    out = OUTPUT_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved: {out}")

# ── Coverage ─────────────────────────────────────────────────────
_plot(
    all_cov,
    "Coverage Rate",
    f"Coverage Rate vs Demand Level\n"
    f"(Fleet: {NUM_MCS_SLOW} Slow + {NUM_MCS_FAST} Fast + {NUM_UAV} UAV)",
    "coverage_vs_demand.png",
    ylim=(0, 1.05),
)

# ── Last Completion Time ─────────────────────────────────────────
_plot(
    all_lct,
    "Last Completion Time (min)",
    f"Last Completion Time vs Demand Level\n"
    f"(Fleet: {NUM_MCS_SLOW} Slow + {NUM_MCS_FAST} Fast + {NUM_UAV} UAV)",
    "last_completion_time_vs_demand.png",
)

print("\nDone.")
