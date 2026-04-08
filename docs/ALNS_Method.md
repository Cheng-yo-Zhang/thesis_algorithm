# ALNS (Adaptive Large Neighborhood Search) 方法說明

## 概述

ALNS 是一種 Destroy-Repair 迭代改善啟發式，搭配 Simulated Annealing (SA) 接受機制與 Adaptive Operator Weight 自適應選擇。其目的是在 Greedy Insertion 產生的初始解上，透過反覆「破壞-修復」來探索鄰域解空間，逐步改善目標函數值。

> **程式碼位置：** `alns.py` — class `ALNSSolver`

---

## 1. 初始解（Greedy Insertion Construction）

ALNS 的 `solve()` 接收由 `greedy_insertion_construction()` 產出的 `initial_solution`，包含：

- 每條 MCS / UAV 路徑的節點序列
- `unassigned_nodes`（Greedy 無法插入的節點）
- `total_cost`（目標函數值）

> **程式碼位置：** `problem.py:421-565` — `greedy_insertion_construction()`

---

## 2. 目標函數

```
Z = α_u · N_miss_urgent  +  α_n · N_miss_normal       (未服務懲罰)
  + β_u · W_urgent_total +  β_n · W_normal_total       (回應時間成本)
  + γ   · D_total                                      (行駛成本)
```

| 參數 | 值 | 說明 |
|------|-----|------|
| `α_u` (ALPHA_URGENT) | 3500.0 | 未服務 urgent 節點懲罰 |
| `α_n` (ALPHA_NORMAL) | 1000.0 | 未服務 normal 節點懲罰 |
| `β_u` (BETA_WAITING_URGENT) | 5.0 | Urgent 用戶等待時間權重 (min) |
| `β_n` (BETA_WAITING_NORMAL) | 0.5 | Normal 用戶等待時間權重 (min) |
| `γ` (GAMMA_DISTANCE) | 1.0 | 行駛距離權重 (km) |

**優先級：** 覆蓋率 >>> 回應時間 >> 行駛距離。少服務一個 urgent 節點的懲罰 (3500) 遠超任何路徑優化能省下的距離或時間。

> **程式碼位置：** `models.py:171-220` — `calculate_total_cost()`

---

## 3. 主迴圈

```
for i in range(5000):                          # ALNS_MAX_ITERATIONS
    1. 複製當前解 → candidate
    2. 決定移除數量 q ∈ [3, 10]
    3. 用輪盤法選一個 Destroy operator + 一個 Repair operator
    4. Destroy: 從 candidate 中移除 q 個節點
    5. Repair:  把移除的節點 + 原本的 unassigned 節點重新插入
    6. Coverage 保護: 若 unassigned 增量 > 2 → 直接拒絕
    7. SA 接受判定: 決定是否接受 candidate
    8. 更新 operator 權重（每 100 次迭代）
    9. 降溫: T *= 0.9995
```

> **程式碼位置：** `alns.py:73-142` — `solve()`

---

## 4. Destroy Operators

### 4.1 Random Removal

- **策略：** 從所有可移除節點中隨機選取 q 個移除
- **目的：** 提供多樣性探索，避免陷入局部最優

> **程式碼位置：** `alns.py:210-216` — `_random_removal()`

### 4.2 Worst Removal

- **策略：** 計算每個節點的 cost 貢獻（saving = 移除後路徑時間減少量），優先移除 saving 最大的節點
- **目的：** 去掉拖累路徑效率的節點，給它們重新分配的機會
- **選擇機制：** 按 saving 降序排序後，使用 `random()^3` 偏斜選擇（偏好 saving 大的但保留隨機性）

> **程式碼位置：** `alns.py:218-241` — `_worst_removal()`

### 4.3 Shaw Removal

- **策略：** 隨機選一個種子節點，移除與它最「相似」的 q 個節點
- **相似度指標：** `relatedness = 曼哈頓距離 + 5.0 × |deadline 差| / 60`
- **目的：** 打散地理 / 時間上的聚類，讓相似節點有機會被重新分配到更好的路徑組合
- **選擇機制：** 按 relatedness 升序排序（越相似越優先），使用 `random()^3` 偏斜選擇

> **程式碼位置：** `alns.py:243-267` — `_shaw_removal()`

### 4.4 Overload Removal

- **策略：** 從節點數最多的路徑中移除約 1/3 的節點
- **目的：** 車隊負載平衡，避免某條路徑過度擁擠
- **處理順序：** 按路徑長度降序處理，直到移除總數達 q

> **程式碼位置：** `alns.py:269-299` — `_overload_removal()`

---

## 5. Repair Operators

### 5.1 Greedy Repair

```
repair_pool = removed_nodes + 原本的 unassigned_nodes
shuffle(repair_pool)                              ← 隨機處理順序

for each node in repair_pool:
    搜尋所有 MCS 路徑的所有位置 → 找 min(delta - bonus + load_penalty)
    若 MCS 全部不可行 且 node 是 urgent → 搜尋 UAV 路徑
    插入最佳位置，或歸入 unassigned
```

**與初始 Greedy Construction 的差異：**

| 面向 | Greedy Construction | Greedy Repair |
|------|---------------------|---------------|
| 處理順序 | EDF (due_date 排序) | Random shuffle |
| Load 機制 | 無 | `load_penalty` 懲罰過載路徑 |
| Repair pool | 僅新請求 + backlog | 移除節點 + 原 unassigned |

**Load Penalty 計算：**

```python
ratio = len(route.nodes) / avg_load
if ratio > 1.0:
    penalty = LAMBDA_BALANCE × 0.1 × (ratio - 1.0)    # LAMBDA_BALANCE = 5.0
```

> **程式碼位置：** `alns.py:304-350` — `_greedy_repair()`、`alns.py:428-440` — `_load_penalty()`

### 5.2 Regret-2 Repair

```
while pool 不為空:
    for each node in pool:
        找所有可行插入位置 → 按 adjusted_cost 排序
        regret = 第二好的 cost - 最好的 cost
        priority = (regret, -due_date, -best_cost)

    選 priority 最高的 node → 插入其最佳位置
    從 pool 移除該 node
```

**核心思想：** 優先插入「如果不現在插就會損失最大」的節點。regret 值高代表這個節點只有一個好位置，必須先搶佔，否則後續可能無處可去。

> **程式碼位置：** `alns.py:352-423` — `_regret2_repair()`

---

## 6. Simulated Annealing 接受機制

```python
delta = candidate.total_cost - current.total_cost

if delta < 0:                                      # 改善 → 必定接受
    accepted = True
elif random() < exp(-delta / T):                    # 允許一定程度惡化
    accepted = True
```

**獎勵分數（用於 Adaptive Weight Update）：**

| 情境 | 分數 | 參數 |
|------|------|------|
| New global best | 33.0 | SIGMA1 |
| Better than current | 9.0 | SIGMA2 |
| SA accepted worse | 3.0 | SIGMA3 |

**溫度排程：**

| 參數 | 值 |
|------|-----|
| 初始溫度 `T₀` | 200.0 |
| 冷卻率 | 0.9995 |
| 最終溫度（5000 次後） | ≈ 16.4 |

> **程式碼位置：** `alns.py:109-123`

---

## 7. Adaptive Weight Update

每 100 次迭代（SEGMENT_SIZE），根據各 operator 在該 segment 內的表現更新其權重：

```python
w_new = w_old × (1 - ρ) + ρ × (score_sum / use_count)
# ρ = 0.1 (ALNS_REACTION_FACTOR)
# 最小權重 = 0.1（避免 operator 完全消失）
```

表現好的 operator（常帶來 global best 或改善）權重增加，被更頻繁選中；表現差的權重下降但不歸零。

> **程式碼位置：** `alns.py:158-163` — `_update_weights()`

---

## 8. Coverage Protection

```python
coverage_loss = len(candidate.unassigned_nodes) - initial_unassigned
if coverage_loss > 2:       # ALNS_MAX_COVERAGE_LOSS
    reject candidate        # 直接拒絕，不進入 SA 判定
```

確保 ALNS 不會為了優化路徑效率而犧牲覆蓋率。即使 SA 溫度高到願意接受惡化解，也不允許 unassigned 節點增加超過 2 個。

> **程式碼位置：** `alns.py:102-107`

---

## 9. Cross-Fleet Local Search（可選）

在 ALNS 主迴圈結束後，可額外執行跨車型的節點交換：

```
for 每對不同車型的 MCS 路徑 (r1, r2):
    for 每對節點位置 (p1, p2):
        嘗試交換 r1[p1] ↔ r2[p2]
        若交換後兩條路徑都可行且 total_time 下降 → 接受
```

**目的：** 讓 MCS-SLOW 和 MCS-FAST 之間的節點分配更合理（例如將 deadline 緊的節點從 SLOW 移到 FAST）。

> **程式碼位置：** `alns.py:455-489` — `cross_fleet_local_search()`

---

## 10. 整體流程圖

```
Greedy Insertion → initial_solution (含 routes + unassigned)
        │
        ▼
   ┌─ ALNS 主迴圈 (5000 iterations) ──────────────┐
   │                                                │
   │  candidate = copy(current)                     │
   │       │                                        │
   │  ┌────▼────┐    ┌──────────┐                   │
   │  │ Destroy │───→│ removed  │                   │
   │  │ (1 of 4)│    │ nodes    │                   │
   │  └─────────┘    └────┬─────┘                   │
   │                      │ + prev unassigned       │
   │                 ┌────▼─────┐                   │
   │                 │  Repair  │                   │
   │                 │ (1 of 2) │                   │
   │                 └────┬─────┘                   │
   │                      │                         │
   │              coverage check                    │
   │              (loss ≤ 2?)                        │
   │                      │                         │
   │               SA acceptance                    │
   │              ┌───────┼────────┐                │
   │          accept   accept   reject              │
   │         (best)   (better)                      │
   │              │       │                         │
   │              ▼       ▼                         │
   │         update weights (every 100 iter)        │
   │         cool down T × 0.9995                   │
   └────────────────────────────────────────────────┘
        │
        ▼
   return best solution
```

---

## 11. ALNS 參數總覽

| 參數 | 值 | 說明 |
|------|-----|------|
| `ALNS_MAX_ITERATIONS` | 5000 | 最大迭代次數 |
| `ALNS_REMOVAL_MIN` | 3 | 每次最少移除節點數 |
| `ALNS_REMOVAL_MAX` | 10 | 每次最多移除節點數 |
| `ALNS_MAX_COVERAGE_LOSS` | 2 | 允許的最大 unassigned 增量 |
| `ALNS_SEGMENT_SIZE` | 100 | 權重更新週期 |
| `ALNS_REACTION_FACTOR` | 0.1 | 權重平滑因子 ρ |
| `ALNS_SIGMA1` | 33.0 | 獎勵：new global best |
| `ALNS_SIGMA2` | 9.0 | 獎勵：better than current |
| `ALNS_SIGMA3` | 3.0 | 獎勵：SA accepted worse |
| `SA_INITIAL_TEMP` | 200.0 | SA 初始溫度 |
| `SA_COOLING_RATE` | 0.9995 | SA 冷卻率 |
| `LAMBDA_BALANCE` | 5.0 | Load-aware penalty 權重 |

---

## 12. 已知結構性問題

ALNS 的兩個 Repair operator（`_greedy_repair` 和 `_regret2_repair`）都繼承了 Greedy Construction 的 **UAV fallback 限制**：只有在所有 MCS 路徑都無法插入時才考慮 UAV（`alns.py:325-328` 和 `377-380`）。這導致：

1. **UAV 在 Destroy-Repair 重組過程中仍被嚴重低估**：即使 Destroy 拆開了路徑，Repair 仍然優先把 urgent 節點塞回 MCS
2. **無法探索 MCS ↔ UAV 的跨車型重分配**：Shaw Removal 可能移除一組相似的 urgent 節點，但 Repair 不會嘗試用 UAV 服務它們（除非所有 MCS 都滿了）
3. **Cross-Fleet Local Search 只處理 MCS 之間的交換**，不包含 MCS ↔ UAV 交換
