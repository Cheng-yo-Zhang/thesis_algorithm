"""
UAV Dispatch Flexibility Comparison — 路徑圖比較
=================================================
比較 MCS+UAV vs MCS-only 在不同 urgent ratio (ρ) 下的路徑差異。
每個 ρ 值產出前 10 slot 的路徑圖，分別存到 with_uav/ 和 mcs_only/ 子資料夾。

輸出到 comparison_uav_flexibility/ 資料夾
"""

import contextlib
import io
from pathlib import Path

from config import Config
from main import run_simulation
from visualization import export_requests_csv, plot_per_slot_mcs_routes


def run_comparison(
    rho_values: list = None,
    output_dir: str = "comparison_uav_flexibility",
):
    if rho_values is None:
        rho_values = [0.5]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fleet_configs = {
        "with_uav": {"NUM_UAV": 20},
        "mcs_only": {"NUM_UAV": 0},
    }

    all_stats = {}

    for rho in rho_values:
        for fc_name, fc_params in fleet_configs.items():
            cfg = Config(
                RANDOM_SEED=42,
                URGENT_RATIO=rho,
                CONSTRUCTION_STRATEGY="deadline",
                NUM_UAV=fc_params["NUM_UAV"],
                MCS_FAST_POWER=50.0,
                UAV_REQUEUE_FOR_MCS=False,
                PLOT_MAX_SLOTS=20,
                T_TOTAL=300,               # 10 slots 生成 + 10 slots cooldown
                GENERATION_END=150,        # t=150 後停止生成，讓 backlog 跑完
                MEASUREMENT_START=0,
                MEASUREMENT_END=150,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                slot_data, fleet, problem, cfg, stats = run_simulation(cfg, verbose=True)
            all_stats[(fc_name, rho)] = stats

            # 輸出子資料夾: with_uav/rho_0.5/ or mcs_only/rho_0.5/
            sub_dir = output_dir / fc_name / f"rho_{rho:.1f}"
            sub_dir.mkdir(parents=True, exist_ok=True)

            with contextlib.redirect_stdout(io.StringIO()):
                export_requests_csv(slot_data, sub_dir / "requests.csv")
                plot_per_slot_mcs_routes(slot_data, cfg, problem, sub_dir)

    # === Summary comparison ===
    lines = [
        "",
        "=" * 80,
        "Summary: MCS+UAV vs MCS-only",
        "=" * 80,
        "",
        f"  {'Config':>10}  {'Service%':>9}  {'Fleet':>6}  {'S':>4}  {'F':>4}  {'U':>4}",
        f"  {'-'*10}  {'-'*9}  {'-'*6}  {'-'*4}  {'-'*4}  {'-'*4}",
    ]
    for rho in rho_values:
        for fc_name in ["with_uav", "mcs_only"]:
            s = all_stats[(fc_name, rho)]
            label = "MCS+UAV" if fc_name == "with_uav" else "MCS-only"
            lines.append(
                f"  {label:>10}  {s['mw_service_rate']:8.1%}  "
                f"{s['mw_fleet_used']:6d}  "
                f"{s['mw_n_slow']:4d}  {s['mw_n_fast']:4d}  {s['mw_n_uav']:4d}"
            )
        lines.append("")

    lines.append("=" * 80)
    summary_text = "\n".join(lines)
    print(summary_text)

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")


if __name__ == "__main__":
    run_comparison(
        rho_values=[0.5],
    )
