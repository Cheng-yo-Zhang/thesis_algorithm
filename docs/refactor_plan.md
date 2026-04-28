# 重構計畫：Static Single-Batch

## 動機

目前程式碼為 rolling-horizon 架構（48hr × 15min slot），但實驗已大量使用 static single-batch（`exp_distance_breakdown`、`exp_fleet_vs_demand`、`exp_route_comparison`、`exp_uav_vs_tw`、`exp_uav_vs_urgent_ratio` 等）。

關鍵指標：
- 核心代碼 ~2,500 行 + 實驗膠水代碼 ~3,100 行（膠水/核心 = 1.24:1）
- `problem.py` 1263 行（god module，混合需求生成 / 評估 / 6 種構造啟發式）
- 12 個實驗腳本大量複製貼上（`ci95`、`make_cfg`、rolling-horizon loop）
- 0% 測試覆蓋率
- `CLAUDE.md` 為空

論文圖表多為 static instance 上的演算法效能比較，rolling-horizon 提供的「動態系統表現」並未在主要圖表中展現，因此可全面退回 static。

## Branch 策略

採「切 branch 不合併、兩條 branch 並存」方案：

```
main                         保留 rolling-horizon 版本（凍結，不再修改）
refactor/static-single-batch 重構成 static 版本（論文主開發線）
```

紀律：
- main 從重構開始後不再有任何 commit（否則兩條 branch 漸行漸遠）
- 論文圖表全部在 `refactor/static-single-batch` 上產出，不混用兩個版本的數據
- 切換 branch 前必須確認當前 branch 工作區是 clean 狀態

## 重構決策摘要

### 已確定保留

- `config.py`：參數中心（精簡至 ~30 個欄位）
- `models.py`：資料結構（`Node`、`Route`、`Solution`、`DistanceMatrix`）
- `problem.py`：保留 `evaluate_route` 與兩種 Greedy 變體（EDF、NN）
- `alns.py`：ALNS 元啟發式
- `simulation.py`：重新定位為「static instance 生成 + 求解 + 路徑資料輸出」
- `experiment.py`：5-6 個實驗集中於單一檔案（待確認）
- `visualization.py`：統一繪圖層（路徑圖 + 指標圖）

### 已確定刪除

| 項目 | 理由 |
|------|------|
| Reserve MCS 機制 | 0 個實驗啟用（`ENABLE_RESERVE_ACTIVATION` 全為 False） |
| `savings_construction` | 0 個實驗呼叫（244 行死碼） |
| `sweep_regret2_construction` | 0 個實驗呼叫（144 行死碼） |
| `UAV_REQUEUE_FOR_MCS` 機制 | rolling-horizon 副產物 |
| `simulation.py:simulate_slot_execution/simulate_dispatch_window` | rolling-horizon 專用 |
| `main.py:run_simulation` | rolling-horizon 主迴圈 |
| `compare_*.py` × 4 | rolling-horizon 重複骨架 |
| `experiment.py` (現有版本)、`fleet_sizing_experiment.py`、`generate_slot_analysis.py`、`smoke_uav_urgent.py`、`_recover_plots.py`、`plot_best_seed.py` | 整合進新框架或刪除 |
| `slot_analysis.md` | 自動產生的 109KB 文件 |

### 待確認決策

1. **Regret-2 不保留？**（目前 5 個 static 實驗使用）
2. **ALNS 初始解統一用 EDF**（影響 5 個實驗數值）
3. **`experiment.py` 單檔，圖片存到results**
4. **`fig_served_vs_completion_time.png` 是新圖**

## 五階段執行計畫

每階段結束後暫停驗證，CSV 結果應與起點一致或變化可解釋。

### 階段 0：清死碼

不改動任何邏輯。CSV 結果應**完全不變**。

| 動作 | 位置 |
|------|------|
| 刪 `_try_reserve_activation` | `problem.py:1197-1246` |
| 刪 `problem.py` 中 reserve 呼叫 | `problem.py:558-561, 613-616` |
| 刪 reserve pool 初始化 | `simulation.py:38-53` |
| 刪 `ENABLE_RESERVE_ACTIVATION`、`RESERVE_MCS_*` | `config.py:142-145` |
| 刪 `VehicleState.is_active` | `models.py` |
| 刪 `savings_construction` | `problem.py:709-951` |
| 刪 `sweep_regret2_construction` | `problem.py:954-1097` |
| 統一 strategy 字串：`"deadline"` / `"alns"` → `"edf"` | 多處實驗腳本 |
| 加上 strategy 字串驗證（unknown value raise ValueError） | `problem.py:461-492` |
| 修正錯誤 docstring | `exp_distance_breakdown.py:8`、`exp_route_comparison.py:81` |

預期影響：核心代碼 -400 行，邏輯無變動。

### 階段 1：拿掉 rolling-horizon

不可逆架構動作。

| 動作 | 影響 |
|------|------|
| 刪 `simulate_slot_execution`、`simulate_dispatch_window` | `simulation.py` 約 -200 行 |
| 刪 `run_simulation` 主迴圈 | `main.py` 約 -250 行 |
| 改寫 `models.py`：移除 `VehicleState.committed_*`、`Route.start_*`、`Node.origin_slot/status='backlog'` | -50 行 |
| 簡化 `config.py`：移除 `T_TOTAL/DELTA_T/MEASUREMENT_*/WARMUP_END/USE_HPP/NHPP_PROFILE/LAMBDA_BAR/USE_FIXED_DEMAND` | 60 → 30 欄位 |
| 加上 `cfg.N_REQUESTS` | 新欄位 |
| 刪 `compare_gi_alns.py`、`compare_gi_nn.py`、`compare_regret2_greedy.py`、`compare_uav_flexibility.py` | 全部 rolling-horizon |
| 刪 `experiment.py` (現有)、`fleet_sizing_experiment.py`、`generate_slot_analysis.py`、`smoke_uav_urgent.py`、`_recover_plots.py`、`plot_best_seed.py` | |
| 改寫 `exp_algorithm_comparison.py` 為 static 版本 | |
| 刪 `slot_analysis.md` | -109KB |
| 刪 UAV requeue 邏輯 | 多處 |

預期影響：總行數 -2,000 至 -2,500。

### 階段 2：簡化 problem.py 構造啟發式

依「待確認決策」的答案決定具體動作：

- **若 Regret-2 保留**：`problem.py` 剩 ~600 行（greedy + regret2 + nn + evaluator）
- **若 Regret-2 刪除**：`problem.py` 剩 ~400 行
- **若 ALNS 初始解統一 EDF**：`alns.py` 移除 `_regret2_repair`，repair operator 簡化

開始前必須跑一次完整實驗，將結果存檔，作為 baseline。

### 階段 3：重組目錄結構

目標結構：

```
thesis_algorithm/
├── CLAUDE.md
├── README.md
├── packages.txt
├── config.py
├── models.py
├── problem.py
├── alns.py
├── simulation.py
├── experiment.py        (或 experiments/ 資料夾)
├── visualization.py
├── tests/
│   ├── test_evaluator.py
│   ├── test_constructors.py
│   └── test_alns.py
├── data/
├── docs/
│   ├── problem_definition.md
│   ├── greedy_insertion_logic.md
│   ├── regret2.md
│   ├── alns_method.md
│   ├── milp_formulation.md
│   └── code_architecture.md
└── results/
    ├── fleet_vs_demand/
    ├── coverage_vs_demand/
    ├── distance_breakdown/
    ├── algorithm_comparison/
    ├── uav_vs_tw/
    ├── uav_vs_urgent_ratio/
    ├── route_comparison/
    └── fleet_sizing/
```

| 動作 | 細節 |
|------|------|
| 合併 5-6 個 `exp_*.py` 進 `experiment.py` | 每個實驗變成 `def run_xxx()` |
| 統一輸出資料夾 | 14+ 個 `results_*/` `comparison_*/` → `results/<exp_name>/` |
| `simulation.py` 重新定位 | instance 工廠 + 求解器 + 路徑資料輸出 |
| 砍 `visualization.py` 的 per-slot 系列 | 619 → ~200 行 |

### 階段 4：黃金測試 + CLAUDE.md

| 檔案 | 內容 |
|------|------|
| `tests/test_evaluator.py` | 給定固定 instance，預期 cost / distance / waiting time（鎖數值）|
| `tests/test_constructors.py` | EDF / NN / Regret-2 在小實例上可行性 |
| `tests/test_alns.py` | 固定 seed 下 ALNS 結果穩定 |
| `CLAUDE.md` | 程式碼地圖（每個檔案職責、如何跑實驗）|

## 規模預期

| 指標 | 目前 | 階段 0 後 | 階段 4 後 |
|------|------|---------|---------|
| 總行數 | ~5,600 | ~5,000 | ~2,500 |
| Python 檔案數 | 27 | 27 | ~11 |
| 最大檔案 | `problem.py` 1263 行 | `problem.py` ~875 行 | `alns.py` ~500 行 |
| 輸出資料夾數 | 14+ | 14+ | 1（含 8 子資料夾） |
| 測試覆蓋率 | 0% | 0% | 核心數值鎖定 |

## 風險與回滾

### 各階段可逆性

| 階段 | 可逆性 | 風險 |
|------|------|------|
| 階段 0 | 高 | 純刪死碼，可隨時 revert |
| 階段 1 | 低 | rolling-horizon 邏輯刪除後復原成本高 |
| 階段 2 | 中 | 啟發式簡化會改變實驗數值 |
| 階段 3 | 中 | 目錄重組影響所有 import |
| 階段 4 | 高 | 純新增測試與文件 |

### 回滾機制

- 切 branch 前在 main 上打 tag `v1-rolling-horizon`（可選）
- 每階段結束 commit，並標註 `git tag stage-N-complete`
- 出問題隨時 `git checkout main` 或 `git revert <commit>`

### 數據連續性驗證

- 階段 0 結束：CSV 結果應與起點完全一致
- 階段 1 結束：static instance 上的 EDF / Regret-2 / ALNS 結果應與目前 static 實驗一致
- 階段 2 開始前：跑完整實驗存 baseline.csv
- 階段 4 結束：跑完整實驗，與 baseline.csv 比對誤差

## 執行順序建議

1. 處理 main 上 deleted PDF
2. 切 `refactor/static-single-batch` branch
3. 階段 0（清死碼）
4. 確認待確認決策 1-5
5. 階段 1（拿掉 rolling-horizon）
6. 階段 2（簡化構造啟發式）
7. 階段 3（重組目錄）
8. 階段 4（測試 + 文件）
