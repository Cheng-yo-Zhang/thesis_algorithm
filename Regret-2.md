# Regret-2 Insertion Heuristic 規格書

## 1. 問題背景

目前的 construction heuristic（`greedy_insertion_construction`）採用 **Deadline-First Greedy Best Insertion**：
- 插入順序：按 slack 排序（固定、靜態）
- 插入位置：對每個節點，遍歷所有路徑的所有位置，選 adjusted_cost 最小的

此做法在車隊緊張時，可能因為「先插入了容易的節點」佔掉了關鍵位置，導致「只有少數可行位置」的節點最終無法被服務。

**Regret-2 的目標**：動態調整插入順序，讓「選擇最少的節點」優先被插入，結構性地降低 unassigned 數量。

---

## 2. 數學定義

### 2.1 符號

| 符號 | 定義 |
|------|------|
| $\mathcal{U}$ | 尚未插入的節點集合 |
| $\mathcal{R}$ | 所有路徑集合（MCS + UAV） |
| $c_{i}^{*}$ | 節點 $i$ 的最佳（最小）insertion cost |
| $c_{i}^{(2)}$ | 節點 $i$ 的次佳 insertion cost |
| $\Delta_i$ | Regret 值 = $c_{i}^{(2)} - c_{i}^{*}$ |

### 2.2 Regret 值計算

對每個未插入節點 $i \in \mathcal{U}$，計算其在所有可行 (route, position) 組合中的 insertion cost，取最小的兩個：

$$
\Delta_i = c_{i}^{(2)} - c_{i}^{*}
$$

- 若節點 $i$ 只有 **1 個**可行位置：$c_{i}^{(2)} = +\infty$，故 $\Delta_i = +\infty$（最高優先）
- 若節點 $i$ 有 **0 個**可行位置：標記為 unassigned，移出 $\mathcal{U}$

### 2.3 選擇規則

每輪迭代選擇 regret 最大的節點插入：

$$
i^{*} = \arg\max_{i \in \mathcal{U}} \Delta_i
$$

**Tie-breaking**：若多個節點 regret 相同，選 $c_{i}^{*}$ 最小的（插入成本最低者）。

插入位置：使用 $c_{i}^{*}$ 對應的 (route, position)。

---

## 3. 演算法流程

```
輸入：active_requests, vehicle_states, dispatch_window_end
輸出：Solution

1. 初始化路徑（與現有 greedy_insertion_construction 相同）
   - 為每台 active 車輛建立空 Route，設定 start_position/time/energy

2. U ← active_requests 的副本（未插入集合）
   unassigned ← []

3. WHILE U 非空:
   3.1 對每個節點 i ∈ U:
       - 掃描所有 (route, pos) 組合
       - 呼叫 incremental_insertion_check(route, pos, node)
       - 收集所有 feasible 的 (adjusted_cost, route_type, route_idx, pos)
       - 排序，取 best_cost (c*) 和 second_best_cost (c(2))
       - 計算 regret_i = c(2) - c*
       - 若無任何可行位置 → 移入 unassigned，從 U 移除

   3.2 從 U 中選 regret 最大的節點 i*
       - Tie-break: regret 相同時選 c* 最小的

   3.3 將 i* 插入其最佳 (route, pos)
       - 呼叫 route.insert_node(pos, node)
       - 呼叫 evaluate_route(route)
       - 設定 node.status = 'assigned'

   3.4 從 U 移除 i*

4. Reserve MCS Activation（與現有邏輯相同）

5. solution.unassigned_nodes = unassigned
   solution.calculate_total_cost(len(active_requests))
   return solution
```

---

## 4. Insertion Cost 定義

沿用現有 `incremental_insertion_check` 的 delta_cost 定義：

```python
delta_cost = candidate.total_time - route.total_time
```

搭配 Type Preference Bonus（沿用現有 `_get_type_preference_bonus`）：

```python
adjusted_cost = delta_cost - bonus
```

**MCS vs UAV 搜尋邏輯**沿用現有的層級式設計：
1. 先搜尋所有 MCS 路徑
2. 僅當 MCS 全部 infeasible 且節點為 urgent 且未被 UAV 服務過時，才搜尋 UAV 路徑

因此每個節點的 regret 計算分兩種情況：
- **有 MCS 可行位置**：regret 只在 MCS 候選中計算（不混入 UAV cost）
- **僅 UAV 可行**：regret 在 UAV 候選中計算

---

## 5. 與現有程式碼的整合

### 5.1 新增方法

在 `ChargingSchedulingProblem` 類別（`problem.py`）中新增：

```python
def regret2_insertion_construction(
    self,
    active_requests: List[Node],
    vehicle_states: List[VehicleState],
    dispatch_window_end: float = None
) -> Solution:
```

### 5.2 可複用的現有元件

| 元件 | 位置 | 用途 |
|------|------|------|
| `incremental_insertion_check()` | problem.py:326 | 計算單一 (route, pos) 的 feasibility + cost |
| `_get_type_preference_bonus()` | problem.py:394 | 車型偏好獎勵 |
| `evaluate_route()` | problem.py | 路徑時間窗評估 |
| `compute_uav_delivery()` | problem.py | UAV 可充電量計算 |
| `_try_reserve_activation()` | problem.py | Reserve MCS 啟用 |
| Route 初始化邏輯 | problem.py:430-458 | 從 VehicleState 建立空路徑 |

### 5.3 Config 整合

在 `config.py` 中擴展 `CONSTRUCTION_STRATEGY` 的選項：

```python
CONSTRUCTION_STRATEGY: str = "regret2"  # "nearest" | "deadline" | "regret2" | "alns"
```

### 5.4 呼叫端修改

在 `main.py` 的策略分支中新增：

```python
if cfg.CONSTRUCTION_STRATEGY == "regret2":
    solution = problem.regret2_insertion_construction(
        active_pool, fleet, dispatch_window_end
    )
```

---

## 6. 內部資料結構

每輪迭代中，對每個未插入節點維護一個候選清單：

```python
@dataclass
class InsertionCandidate:
    node: Node
    best_cost: float           # c*
    best_route_type: str       # 'mcs' | 'uav'
    best_route_idx: int
    best_position: int
    second_best_cost: float    # c(2)
    regret: float              # c(2) - c*
    feasible_count: int        # 可行位置數量
```

---

## 7. 效能考量

### 7.1 時間複雜度

| | Greedy Best Insertion | Regret-2 |
|--|--|--|
| 外層迴圈 | O(N) 個節點，固定順序 | O(N) 輪，每輪插入 1 個 |
| 每輪工作量 | 1 個節點 × R 路徑 × P 位置 | **U 個節點** × R 路徑 × P 位置 |
| 總複雜度 | O(N × R × P) | O(N² × R × P) |

其中 N = 節點數，R = 路徑數，P = 平均路徑長度。

### 7.2 實際影響

以目前的參數估算（`FIXED_DEMAND_PER_SLOT=3`）：
- N=3, R=60, P≈1-2
- Greedy: 3 × 60 × 2 = 360 次 insertion check
- Regret-2: (3+2+1) × 60 × 2 = 720 次 insertion check

差距僅 2 倍，在此規模下可忽略。

即使在 NHPP 高峰（N≈15-20），Regret-2 約為 Greedy 的 N/2 ≈ 10 倍，仍在毫秒級，不構成瓶頸。

### 7.3 可選優化：增量更新

插入一個節點後，只有**同一條路徑上的候選**需要重算。其他路徑的 insertion cost 不受影響。可用 dirty flag 標記需要重算的路徑，避免全量重算。

---

## 8. 正確性保證

- **覆蓋率不低於 Greedy**：Regret-2 優先保護「沒退路的節點」，在車隊緊張時 unassigned 數量 ≤ Greedy
- **可行性保證不變**：每次插入仍經過 `incremental_insertion_check` 的完整時間窗與容量檢查
- **MCS/UAV 層級不變**：UAV 仍只在 MCS 全部 infeasible 時啟用
- **Type Preference Bonus 不變**：MCS-SLOW 優先邏輯保持一致

---

## 9. 驗證方式

1. **單元測試**：構造一個車隊緊張的 scenario（例如 2 台 MCS、5 個請求），確認 Regret-2 比 Greedy 少丟人
2. **回歸測試**：在車隊充裕情境下（現有參數），確認 Regret-2 與 Greedy 的 coverage 相同（均為 100%）
3. **Fleet sizing 實驗**：逐步縮減車輛數，比較兩種策略的 coverage 曲線分離點
