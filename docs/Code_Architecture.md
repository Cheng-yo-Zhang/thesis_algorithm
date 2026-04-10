# 程式碼分工說明（Code Architecture）

> **專案**：HE-EVRPTW-SD — Heterogeneous EV Routing Problem with Time Windows and Split Deliveries
> **求解方法**：Construction Heuristics（Greedy / Regret-2 / Sweep / NN）+ Adaptive Large Neighborhood Search (ALNS)
> **架構特性**：Rolling-Horizon 動態派遣、MCS（地面充電車）+ UAV（空投無人機）異質車隊

本文件整理 `thesis_algorithm/` 根目錄下所有 Python 程式碼的職責分工，依功能分層說明，作為日後開發、論文寫作、以及新成員上手的參考。

---

## 1. 整體分層架構

```
┌────────────────────────────────────────────────────────┐
│  Layer 5: Experiment Drivers / Comparison Scripts      │
│  exp_*.py, compare_*.py, fleet_sizing_experiment.py    │
│  experiment.py, generate_slot_analysis.py              │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  Layer 4: Simulation Orchestration                     │
│  main.py（rolling-horizon 主迴圈）                      │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  Layer 3: Algorithm / Mechanics                        │
│  alns.py（metaheuristic）  simulation.py（dispatch）    │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  Layer 2: Problem Model & Construction Heuristics      │
│  problem.py                                            │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│  Layer 1: Data Structures & Configuration              │
│  models.py            config.py                        │
└────────────────────────────────────────────────────────┘
```

依賴方向由上往下：上層只能 import 下層，下層不會反向依賴上層。

---

## 2. 基礎層：資料結構與設定

### 2.1 `config.py` — 集中化參數設定
- **職責**：以 `Config` dataclass 統一管理約 52 個實驗參數，避免散落於各檔案。
- **主要內容**：
  - **到達流程**：HPP / NHPP 切換、24 小時時變到達率 profile（`get_arrival_rate_profile()`）。
  - **車隊組成**：MCS-Slow、MCS-Fast、UAV 數量；備援車輛選項。
  - **車輛物理參數**：電池容量、行駛能耗率、慢/快充功率、UAV 載能 $Q^{uav}$、目標 SOC $\sigma$。
  - **目標函數權重**：$\alpha_u, \alpha_n$（未服務懲罰）、$\beta_u, \beta_n$（等待懲罰）、距離成本。
  - **ALNS 超參數**：迭代數、SA 溫度、操作子權重更新率。
- **被誰使用**：所有其他檔案。
- **層級**：Layer 1（無任何專案內部依賴）。

### 2.2 `models.py` — 解的資料模型
- **職責**：定義所有跨模組共用的資料類別，是解（Solution）與車隊狀態（VehicleState）的資料載體。
- **主要類別**：
  | 類別 | 用途 |
  |---|---|
  | `Node` | 充電請求節點：座標、能量需求 $D_i$、釋放/截止時間 $r_i, \bar d_i$、緊急類別、初始 SOC |
  | `VehicleState` | 滾動視界中車輛的當前狀態：位置、可用時間、剩餘能量、已 commit 節點（凍結前綴） |
  | `Route` | 單一車輛路徑：節點序列、抵達/離開時間、距離、等待時間、可行性旗標 |
  | `Solution` | 多車路徑的容器，計算三項目標（覆蓋率、等待懲罰、行駛成本） |
  | `DistanceMatrix` | 預先計算的距離快取（MCS 用 Manhattan，UAV 用 Euclidean） |
- **依賴**：`config`。
- **層級**：Layer 1。

---

## 3. 問題模型層

### 3.1 `problem.py` — 問題實例與構造式啟發法
- **職責**：封裝 EVRPTW 問題的所有「靜態」邏輯——產生請求、檢查可行性、五種構造式啟發法的實作。
- **核心類別**：`ChargingSchedulingProblem`
- **核心方法**：
  - **請求生成**：`generate_requests()`（HPP / NHPP Poisson 或固定速率，含緊急/一般分類）
  - **構造式啟發法**：
    1. `greedy_insertion_construction()` — EDF / Slack-based 貪婪插入
    2. `regret2_insertion_construction()` — Regret-2 後悔值插入
    3. `savings_construction()` — Clarke-Wright 節省法
    4. `sweep_regret2_construction()` — Sweep 角度分群 + Regret-2
    5. `nearest_neighbor_construction()` — 最近鄰鏈
  - **可行性與評估**：`evaluate_route()`、`try_insert_node()`、`calculate_distance()`、`calculate_travel_time()`、`calculate_charging_time()`
- **依賴**：`config`、`models`。
- **被誰使用**：`alns.py`、`main.py`、`simulation.py`、所有 experiment scripts。
- **層級**：Layer 2。

---

## 4. 演算法與機制層

### 4.1 `alns.py` — Adaptive Large Neighborhood Search 求解器
- **職責**：在構造式解之上做迭代改善，搭配 Simulated Annealing 接受準則與自適應操作子權重。
- **核心類別**：`ALNSSolver`
- **核心方法**：
  - `solve()` — ALNS 主迴圈（destroy → repair → SA accept → weight update）
  - **Destroy 操作子**：`_random_removal`、`_worst_removal`、`_shaw_removal`（相關性移除）、`_overload_removal`
  - **Repair 操作子**：`_greedy_repair`、`_regret2_repair`
  - **自適應機制**：`_select_operator`、`_update_weights`（按 reward 動態調整操作子被選機率）
- **依賴**：`config`、`models`、`problem`。
- **被誰使用**：`main.py`、所有需要做精緻化的 experiment / comparison scripts。
- **層級**：Layer 3。

### 4.2 `simulation.py` — 車隊執行模擬
- **職責**：在 rolling-horizon 框架中模擬車輛真正「跑」一段路線：時間推進、能量扣除、節點凍結（committed）、UAV 充電策略。
- **核心函式**：
  - `initialize_fleet()` — 依 `Config` 建立 `VehicleState` 池（含可選備援車）
  - `simulate_dispatch_window()` — 執行該 slot 派出的所有 Route，回填車輛狀態與節點完成時間
  - `simulate_slot_execution()` — 舊版單 slot 介面（已逐步被取代）
- **依賴**：`config`、`models`、`problem`。
- **被誰使用**：`main.py`、`compare_gi_alns.py`、`compare_gi_nn.py`、`generate_slot_analysis.py`、`exp_coverage_vs_demand.py`、`exp_fleet_vs_demand.py`。
- **層級**：Layer 3。

---

## 5. 模擬編排層

### 5.1 `main.py` — Rolling-Horizon 主控
- **職責**：串接所有下層模組，實作完整的滾動視界派遣邏輯。是「跑一次完整模擬」的單一入口。
- **核心函式**：`run_simulation(cfg)`
- **每個 slot 的執行流程**：
  1. 依到達率產生新請求並併入 backlog
  2. 過濾過期請求（deadline expiry）
  3. 呼叫指定的 construction heuristic 建構初始解
  4. （可選）呼叫 `ALNSSolver.solve()` 改善
  5. 透過 `simulate_dispatch_window()` 推進車隊狀態
  6. 收集每 slot 的 KPI（coverage、等待時間、成本、車隊使用率）
- **依賴**：`config`、`models`、`problem`、`simulation`、`alns`。
- **被誰使用**：所有 experiment / comparison 腳本（作為單次 run 的呼叫端點）。
- **層級**：Layer 4。

---

## 6. 實驗驅動層

> 此層的腳本不含演算法新邏輯，皆是 `main.run_simulation` 或 `problem` 構造方法的封裝，用來重現論文圖表。

### 6.1 敏感度與比較實驗

| 檔案 | 對應論文圖/分析 | 變動軸 | 比較對象 |
|---|---|---|---|
| `experiment.py` | ρ 敏感度 | 緊急請求比例 ρ | 服務率、車隊使用率（多 seed CI） |
| `exp_algorithm_comparison.py` | Fig. 6：λ 敏感度 | 到達率 λ | Greedy-EDF / Regret-2 / Regret-2+ALNS |
| `exp_coverage_vs_demand.py` | Coverage curve | 固定車隊下的請求數 [10–60] | 三種演算法 |
| `exp_fleet_vs_demand.py` | 車隊規模曲線 | 請求數 [10–80] | 各構造式所需最小車隊 |
| `exp_route_comparison.py` | 路徑視覺化 | 單派遣 instance | Greedy / Sweep+R2 / ALNS 三圖並列 |
| `fleet_sizing_experiment.py` | 參數化車隊掃描 | 單一車型數量 | 三種構造式 |

### 6.2 對照實驗（Comparison Scripts）

| 檔案 | 對照主題 |
|---|---|
| `compare_gi_alns.py` | Greedy Insertion vs GI + ALNS（驗證 ALNS 增益） |
| `compare_gi_nn.py` | Greedy Insertion vs Nearest-Neighbor |
| `compare_regret2_greedy.py` | Greedy-EDF / Greedy-Slack / Regret-2 三策略並列 |
| `compare_uav_flexibility.py` | MCS+UAV 與 MCS-only 在不同 ρ 下的比較（UAV 彈性消融） |

### 6.3 報表與輔助

| 檔案 | 用途 |
|---|---|
| `generate_slot_analysis.py` | 產生 `slot_analysis.md`：前 10 個 slot 的逐節點派遣決策、可行性、KPI 表 |

---

## 7. 視覺化與輸出

### 7.1 `visualization.py`
- **職責**：所有繪圖與 CSV 輸出的集中模組。
- **核心函式**：
  - `export_requests_csv()` — 請求屬性匯出
  - `plot_per_slot_distribution()` — 每 slot EV 空間散佈圖
  - `plot_per_slot_mcs_routes()` — MCS / UAV 軌跡圖（含充電模式上色）
  - `print_terminal_report()` — 終端統計摘要
- **依賴**：`config`、`problem`。

### 7.2 `plot_best_seed.py`
- **職責**：從現成 CSV 結果中挑出每個 demand level 的最佳 seed，繪製 Greedy vs Regret-2+ALNS 並列圖（避免重跑模擬）。

---

## 8. 目錄結構

| 目錄 | 內容 |
|---|---|
| `config/` | 預留 JSON/YAML 設定檔（目前未使用，所有設定走 `config.py`） |
| `data/` | 預留輸入資料集（目前皆為合成生成，未使用） |
| `docs/` | 文件：`ALNS_Method.md`、`MILP_Formulation.md`、本檔 `Code_Architecture.md` |
| `output/` | 通用輸出根目錄，內含 `route_comparison/` 子資料夾 |
| `results_coverage_vs_demand/` | `exp_coverage_vs_demand.py` 產出 |
| `results_fleet_vs_demand/` | `exp_fleet_vs_demand.py` 產出 |
| `comparison_gi_alns/` 等 | 各 `compare_*.py` 對應產出 |
| `fleet_sizing_results/` | `fleet_sizing_experiment.py` 產出 |
| `image/`、`temp_vis_test/` | 暫存與圖檔素材 |

---

## 9. 依賴關係圖（簡化版）

```
config.py
   │
   ▼
models.py
   │
   ▼
problem.py ────────────────┐
   │                       │
   ├──► simulation.py      │
   │                       │
   └──► alns.py ◄──────────┘
            │
            ▼
        main.py
            │
   ┌────────┼────────┬──────────────┬──────────────┐
   ▼        ▼        ▼              ▼              ▼
experiment exp_*  compare_*  fleet_sizing_*  generate_slot_*
```

`visualization.py` 為旁路工具，被 `main.py` 與多數 `compare_*` 腳本呼叫。

---

## 10. 模組職責一覽表

| 模組 | 層級 | 類別 | 一句話定位 |
|---|---|---|---|
| `config.py` | L1 | 設定 | 集中所有實驗參數，單一事實來源 |
| `models.py` | L1 | 資料結構 | Node / Route / Solution / VehicleState 定義 |
| `problem.py` | L2 | 問題模型 | EVRPTW 實例 + 五種構造式啟發法 |
| `simulation.py` | L3 | 機制 | Rolling-horizon 中車隊狀態的真實推進 |
| `alns.py` | L3 | 演算法 | ALNS metaheuristic 求解器 |
| `main.py` | L4 | 編排 | 完整 rolling-horizon 模擬入口 |
| `experiment.py` | L5 | 實驗 | ρ 敏感度分析 |
| `exp_algorithm_comparison.py` | L5 | 實驗 | λ 敏感度（Fig. 6） |
| `exp_coverage_vs_demand.py` | L5 | 實驗 | 固定車隊覆蓋率曲線 |
| `exp_fleet_vs_demand.py` | L5 | 實驗 | 最小車隊規模曲線 |
| `exp_route_comparison.py` | L5 | 實驗 | 三演算法路徑視覺化 |
| `fleet_sizing_experiment.py` | L5 | 實驗 | 單車型參數化掃描 |
| `compare_gi_alns.py` | L5 | 對照 | GI vs GI+ALNS |
| `compare_gi_nn.py` | L5 | 對照 | GI vs NN |
| `compare_regret2_greedy.py` | L5 | 對照 | 三構造式並列 |
| `compare_uav_flexibility.py` | L5 | 對照 | UAV 消融 |
| `generate_slot_analysis.py` | L5 | 報表 | 逐 slot Markdown 報告 |
| `visualization.py` | — | 工具 | 繪圖與 CSV 輸出 |
| `plot_best_seed.py` | — | 工具 | Best-seed 對比繪圖 |

---

## 11. 開發守則建議

1. **新增實驗腳本** → 應放在 Layer 5，呼叫 `main.run_simulation()` 或直接使用 `problem` 的構造方法，**不應**重新實作 dispatch 邏輯。
2. **新增 ALNS 操作子** → 放入 `alns.py` 的 `ALNSSolver` 內，並在權重表註冊。
3. **新增約束或可行性檢查** → 修改 `problem.py` 的 `evaluate_route()` / `try_insert_node()`，避免散落在演算法層。
4. **新增參數** → 一律加入 `config.py` 的 `Config` dataclass，並提供合理預設值；不要在實驗腳本內 hard-code。
5. **新增繪圖** → 集中在 `visualization.py`，保持實驗腳本只負責「跑模擬 + 寫 CSV」，繪圖另呼叫工具函式。
