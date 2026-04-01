# Greedy Insertion Construction 邏輯說明

## 函數位置
`problem.py` — `greedy_insertion_construction()` (L421-565)

---

## Phase 0：初始化路徑 (L442-458)

為每台 **active** 車輛建立空路徑，繼承車輛狀態：
- `start_position`：車輛當前位置
- `start_time`：車輛可用時間 (`available_time`)
- `start_energy`：車輛剩餘能量
- `is_deployed`：是否有 committed nodes（前一 slot 的未完成服務）

MCS 路徑加入 `solution.mcs_routes`，UAV 路徑加入 `solution.uav_routes`。

---

## Phase 1：排序請求 (L460-492)

依 `CONSTRUCTION_STRATEGY` 決定 request 處理順序：

| 策略 | 排序依據 | 說明 |
|------|---------|------|
| `edf` | `(due_date, -demand)` | Earliest Deadline First |
| `slack` | `(slack, due_date)` | slack = deadline − (dispatch_time + min_travel + min_service) |
| `nearest` | Manhattan distance to depot | 離 depot 最近優先 |
| `deadline` (fallback) | `due_date` | 僅按 deadline 排序 |

- `slack` 策略中，`min_travel` 取所有 active 車輛位置到該 request 的最短 Manhattan 距離
- `min_service` 取最大充電功率下的最短充電時間

---

## Phase 2：逐一插入 — Type-Flexible Unified Cost Dispatching (L496-555)

對排序後的每個 request，依序執行：

### Step 1 — 搜尋所有 MCS 路徑 (L503-513)

同時搜尋 MCS-SLOW 和 MCS-FAST 的所有路徑與所有插入位置：

```
for each MCS route:
    for each position pos in [0, ..., len(route.nodes)]:
        (feasible, delta_cost) = incremental_insertion_check(route, pos, node)
        if feasible:
            bonus = type_preference_bonus(route.vehicle_type, node)
            adjusted_cost = delta_cost - bonus
            if adjusted_cost < min_adjusted_cost:
                update best insertion = (route, pos)
```

### Step 2 — 搜尋 UAV 路徑 (L516-530)

**僅在所有 MCS 都不可行時**才搜尋 UAV，且需滿足：
- `best_route_type is None`（沒有任何 MCS 可行插入）
- `node.node_type == 'urgent'`
- `compute_uav_delivery(node) > 0`（UAV 可交付能量）
- `not node.uav_served`（未被 UAV 服務過）

搜尋邏輯同 Step 1。

### Step 3 — 執行插入 (L532-555)

- 將 node 插入到最佳位置
- `evaluate_route()` 重新驗證整條路徑的時間窗
- 若驗證失敗：回滾（`remove_node` + 重新 evaluate），node 加入 `unassigned`

---

## Phase 3：Reserve Activation (L558-561)

- 條件：有 unassigned 且 `ENABLE_RESERVE_ACTIVATION=True`
- 嘗試啟用備用 MCS 車輛來服務剩餘未分配的 request

---

## 核心子函數

### `incremental_insertion_check(route, pos, node)` (L326-354)

回傳 `(feasible: bool, delta_cost: float)`

1. **容量檢查**：
   - MCS：`route.total_demand + node.demand ≤ capacity`（除非 `MCS_UNLIMITED_ENERGY`）
   - UAV：累計交付量 ≤ `UAV_DELIVERABLE_ENERGY`；已被 UAV 服務過的 node 不可再服務
2. **時間窗檢查**：
   - 複製路徑 → 插入 node → `evaluate_route()` 驗證所有節點的 departure ≤ due_date
3. **Cost**：
   - `delta_cost = candidate.total_time - route.total_time`（插入後路徑總時間增量）

### `_get_type_preference_bonus(vehicle_type, node)` (L394-418)

用於引導車輛類型選擇的 soft preference：

| 車輛類型 | Bonus | 說明 |
|---------|-------|------|
| MCS-SLOW | `(demand/SLOW_POWER - demand/FAST_POWER) × 60` min | Slow 與 Fast 充電時間差，讓 Slow 在 adjusted_cost 上與 Fast 對齊，優先使用經濟的 Slow |
| MCS-FAST | 0 | 無額外獎勵 |
| UAV | 10 min（僅 urgent 且 slack ≤ 15min） | 極緊迫的 urgent 才給 UAV 獎勵 |

### `try_insert_node(route, node, position)` (L285-315)

- 若未指定 position：遍歷所有位置，找 delta_cost 最小的可行位置
- 同 cost 時以 `_calculate_insertion_delta_dist`（距離增量）作 tiebreak

---

## 演算法特性總結

- **Sequential Greedy**：按排序逐一處理 request，每次找全局最小 adjusted_cost 的 (route, position) 插入
- **MCS 優先於 UAV**：UAV 僅作為 MCS 不可行時的 fallback
- **Soft Type Preference**：透過 bonus 機制偏好 MCS-SLOW > MCS-FAST > UAV，而非 hard priority
- **Cost Metric**：路徑總時間增量（delta route time），而非距離
