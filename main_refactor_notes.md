# main.py 改造備忘錄（Solomon-R2 / 25 nodes baseline）

此文件彙整你目前 `main.py`（/mnt/data/main.py）要從「**單一 MCS 混合快/慢充**」改成「**Fast-MCS / Slow-MCS 拆兩種車隊 + UAV event-driven**」所需的**關鍵邏輯修正點**，並對應到程式中的函式/段落（含行號）。

---

## 0. 你目前已定案的 baseline（本文件以此為準）

1) **充電量固定**：每個 customer 固定補能 `E_i = 20 kWh`（Solomon `DEMAND` 不再視為充電量）。  
2) **節點數**：先跑 **25 customers（不含 depot）** 做 debug/調參。  
3) **Fleet size（N=25）**：  
   - Slow-MCS：4 台（1 port）  
   - Fast-MCS：1 台  
   - UAV：1 台  
4) **快/慢充需求定義**：依時間窗寬度 `u_i = due_date - ready_time` 排序，前 20%（最短者）標記為 **urgent / fast-demand**。  
5) **UAV 可服務集合**：用距 depot 距離的分位數（建議 80th percentile）判定「偏遠」→ `uav_eligible`.

---

## 1. 目前 main.py 的核心問題（為什麼一定要改）

### 1.1 “混合 MCS” 邏輯仍存在
`get_charging_profile()` 目前用 `node_type` 決定 MCS 的快/慢充（urgent→FAST、normal→SLOW），等於「同一台車會切換功率」：  
- 位置：`get_charging_profile()`（約 L755-L770）

### 1.2 充電時間仍用 Solomon DEMAND 當 kWh
`load_data()` 直接把 CSV 的 `DEMAND` 寫入 `Node.demand`，而充電時間計算使用 `node.demand`：  
- 位置：`load_data()`（約 L575-L675）與 `get_charging_profile()`（約 L755-L770）

### 1.3 時間窗目前同時限制「開始」與「完成」
`evaluate_route()` 與 `incremental_insertion_check()` 都檢查 `departure_time <= due_date`（finish-time hard TW），在你慢充（20kWh @ 11kW）下會非常容易 infeasible：  
- 位置：`evaluate_route()`（約 L819-L833）  
- 位置：`incremental_insertion_check()`（約 L996-L1007）

### 1.4 Fleet size 不固定：會動態“開新車”
建構解與 ALNS repair 都會在插不進去時「動態新增 MCS/UAV route」，導致你設定的 fleet size 失真：  
- 位置：`parallel_insertion_construction()` Phase 2.5（約 L1369-L1401）  
- 位置：`_greedy_insertion()`（約 L1827-L1861）  
- 位置：`_regret2_insertion()`（約 L1928 起）

### 1.5 clustering 自動選 K（Silhouette），且對所有 customers 聚類
`cluster_customers()` 會自動選 K，且把 urgent/normal 全部丟去聚類，等於你「slow-MCS=4」會被 clustering 覆蓋/破壞：  
- 位置：`cluster_customers()`（約 L1153-L1242）

### 1.6 MCS “到達≠立刻充電”尚未落地
`service_start = max(arrival, ready_time)` 沒有加上 **setup/connect/disconnect**，你提到的工作時間需求尚未反映在可行性檢查與等待時間計算中：  
- 位置：`evaluate_route()`（約 L816-L827）  
- 位置：`incremental_insertion_check()`（約 L989-L1006）  
- 位置：`_forward_propagate_check()`（約 L1114-L1139）

---

## 2. 必改項目清單（照這份做，改完就會符合你的新設定）

以下分為「資料層/規則層/路由層/演算法層」四類。

---

## 2.1 資料層（Data / Config / Class）

### (A) 修正 `ChargingSchedulingProblem.__init__` 的 MCSConfig 錯誤
目前 `ChargingSchedulingProblem.__init__` 的型別註解與預設使用 `MCSConfig`，但本檔案沒有 `MCSConfig` class，會造成執行錯誤：  
- 位置：`ChargingSchedulingProblem.__init__`（約 L554-L565）

**建議修正（兩種擇一）：**
1. **最小改動**：新增 `@dataclass class MCSConfig`（保留舊 API），內含 `SPEED, CAPACITY, POWER_FAST, POWER_SLOW`。  
2. **推薦做法**：改 `__init__` 參數為 `mcs_fast_config: FastMCSConfig` 與 `mcs_slow_config: SlowMCSConfig`，並存成 `self.mcs_fast / self.mcs_slow`。

> 後續所有 `self.mcs.xxx` 會改成依 `vehicle_type` 讀 `self.mcs_fast.xxx` 或 `self.mcs_slow.xxx`。

### (B) Route.vehicle_type 擴充成三類
目前 Route 只接受 `'mcs'` 或 `'uav'`：  
- 位置：`Route.__init__`（約 L97-L113）

改成：`'mcs_fast' | 'mcs_slow' | 'uav'`。

### (C) Solution 結構：允許區分快慢 MCS
目前 `Solution` 只有 `mcs_routes` 與 `uav_routes`：  
- 位置：`Solution.__init__`（約 L183-L216）

**最小改動建議**：維持 `mcs_routes` 但每條 route 用 `vehicle_type` 區分快/慢；提供 helper：
- `get_mcs_fast_routes()`
- `get_mcs_slow_routes()`

（或改成三個 list：`mcs_fast_routes / mcs_slow_routes / uav_routes`，但改動面更大。）

---

## 2.2 規則層（Demand / Node type / UAV eligible）

### (D) 固定充電量 20 kWh（不要再用 Solomon DEMAND 當 kWh）
位置：`load_data()`（約 L648-L675）

**建議做法（最小改動）**：
- `load_data()` 完成後：對所有 customer（id != 0）設定 `node.demand = 20.0`（depot/centroid 仍為 0）。  
這樣你不用立刻新增 `energy_kwh` 欄位。

（更乾淨的做法是新增 `Node.energy_kwh`，但會波及較多地方。）

### (E) urgent = 最短時間窗前 20%（你要的規則）
位置：`assign_node_types()`（約 L688-L710）

目前 `assign_node_types()` 幾乎只支援「讀 CSV 的 NODE_TYPE」。你要加一個分支：

- `use_csv_types=False` 時：
  1) 取 customers（不含 depot）
  2) 計算 `tw_width = due - ready`
  3) 排序取前 `ceil(0.2 * N)` → 設 `node.node_type='urgent'`，其餘 `'normal'`
  4) 同步更新 `self.urgent_indices`

### (F) UAV eligible set（距離分位數）
新增：`self.uav_eligible_ids: set[int]`，建議在 `load_data()` 後或 `assign_node_types()` 後計算。

計算方式：
- `dist_i = euclidean(depot, i)`
- `D_thr = quantile(dist_i, q=0.8)`（你可配置 q）
- `eligible = {i | dist_i >= D_thr}`

**落地點（必加硬限制）：**
- `incremental_insertion_check()`：若 `vehicle=='uav'` 且 `node.id not in eligible` → 直接 return infeasible  
- `parallel_insertion_construction()` / repair operators：在嘗試 UAV insertion 前先檢查 eligible（節省計算）

---

## 2.3 路由層（Travel time / Service time / Time-window semantics）

### (G) travel time / distance：支援 mcs_fast/mcs_slow
位置：
- `calculate_travel_time()`（約 L726-L749）
- `calculate_distance()`（約 L711-L724）

目前 `vehicle=='mcs'` 才走曼哈頓。改成：
- `vehicle.startswith('mcs')` 走曼哈頓
- `vehicle=='uav'` 走歐式

同時速度取值：
- mcs_fast：`self.mcs_fast.SPEED`
- mcs_slow：`self.mcs_slow.SPEED`  
（就算兩者數值相同，也要以車種取值，避免後續擴展再大改。）

### (H) “到達≠立刻充電”：把作業時間放入 service start / finish
你需要新增（或從 config 引入）：
- `ARRIVAL_SETUP_MIN`
- `CONNECT_MIN`
- `DISCONNECT_MIN`
- （UAV 另含 `TAKEOFF_LAND_OVERHEAD_MIN`，你已經有）

**將以下公式一致套用到：**
- `evaluate_route()`
- `incremental_insertion_check()`
- `_forward_propagate_check()`

建議統一用：

1) `arrival_time = current_time + travel_time`
2) `service_start = max(node.ready_time, arrival_time + setup_k + connect_k)`
3) `charge_time = (20 / power_k) * 60`
4) `departure_time = service_start + charge_time + disconnect_k`

> 注意：你現在的程式把 UAV 起降時間加在 travel_time（L746-L749）。只要你保持一致也可以，但最乾淨做法是把“起降/落地/接駁”放在 service overhead，而不是每段飛行都加。

### (I) 時間窗語意：改成 start-time hard TW（baseline 建議）
你現在同時檢查：
- `service_start <= due_date`（OK）
- `departure_time <= due_date`（建議移除）

位置：
- `evaluate_route()`（約 L819-L833）
- `incremental_insertion_check()`（約 L996-L1007）
- `_forward_propagate_check()`（需同步改 departure propagation）

**改法：**
- baseline：只保留 `service_start <= due_date`
- 若你要更嚴格：把 `departure_time > due_date` 當 tardiness penalty（soft TW），不要直接 infeasible

---

## 2.4 演算法層（Pre-positioning / clustering / fixed fleet / ALNS repair）

### (J) clustering 只做給 slow-demand，且 K 固定 = NUM_SLOW_MCS
位置：`cluster_customers()`（約 L1153-L1242）

你現在：
- 對全部 customers 聚類（含 urgent）
- 用 silhouette 自動選 K
- Phase 1：每個 cluster 開一台 MCS（等於 fleet size 不固定）

**改法：**
1) `cluster_customers(k: int, nodes: List[Node])`：外部傳入固定 K  
2) nodes 僅使用 `normal`（slow-demand）集合  
3) centroid nodes 數量 = `NUM_SLOW_MCS`（N=25 baseline → 4）
4) 不要讓 `cluster_customers()` 自動改變 fleet size

### (K) Pre-positioning：只對 Slow-MCS 路徑設 `start_node=centroid`
位置：`evaluate_route()`（約 L796-L804）與 `parallel_insertion_construction()` Phase 1（約 L1290-L1299）

改成：
- `mcs_slow` route 才使用 `start_node=centroid`（預部署）
- `mcs_fast` route 建議 `start_node=None`（在 depot 待命或另定 staging 點）

### (L) 固定 fleet size：禁止動態開新車
位置（必改）：
- `parallel_insertion_construction()` 的 2.5（約 L1369-L1401）
- repair operators：`_greedy_insertion()`（約 L1827-L1861）、`_regret2_insertion()`（約 L1928 起）
- destroy operator `_worst_removal()`（約 L1740-L1744）也要支援三種 vehicle_type

**做法（推薦）：**
新增 FleetConfig（或放在 ProblemConfig）：
- `NUM_MCS_SLOW = 4`
- `NUM_MCS_FAST = 1`
- `NUM_UAV = 1`

建構解時先建立固定數量 routes：
- 建立 4 條 `mcs_slow`（各自 start_node=centroid）
- 建立 1 條 `mcs_fast`
- 建立 1 條 `uav`

之後「插不進去」的節點：
- 不允許開新 route → 放進 `solution.unassigned_nodes`
- 由 `PENALTY_UNASSIGNED` 懲罰（Solution 已有）

### (M) event-driven：UAV 不分群，僅在事件發生時 dispatch
你目前的 `parallel_insertion_construction()` 是一次性構造解（離線）。要做 event-driven，最小改法是：

- 把 customer 以 `ready_time` 排序（不是 due_date EDD），視為事件序列
- 逐筆處理每個 request（t = ready_time）時才嘗試 insertion
- UAV 的 candidate set：`urgent AND uav_eligible` 才允許嘗試 UAV

> 你可以先保留 EDD 作 tie-breaker，但主排序應改為 ready_time（才是 event-driven）。

---

## 3. 你改完後，應該具備的行為（驗收清單）

### 3.1 baseline 行為
- **所有 customer 的能量需求都等於 20 kWh**（depot/centroid = 0）
- urgent 節點數 = `ceil(0.2 * N)`（N=25 → 5）
- UAV 只會服務：`urgent AND uav_eligible`
- Slow-MCS 只服務：normal（baseline）
- Fast-MCS 只服務：urgent（baseline）

### 3.2 不會偷跑的行為
- 無論插入失敗與否，route 數量永遠固定：
  - slow routes = 4
  - fast routes = 1
  - uav routes = 1

### 3.3 時間一致性
- MCS/UAV 到達後需經過 setup/connect 才能開始充電  
- time-window 只限制「開始服務」；慢充完成可超過 due_date（或以 penalty 計）

---

## 4. 最小改動的建議實作順序（照做比較不痛）

1) **先修 `MCSConfig` 引用錯誤**（讓程式能跑）  
2) Route.vehicle_type 改成三類；改 travel/distance 分流  
3) 固定 20 kWh：在 `load_data()` 後覆寫 customers 的 `demand=20`  
4) urgent=TW 前 20%：補完 `assign_node_types(use_csv_types=False)`  
5) UAV eligible set：加 set 並在 insertion check 加硬限制  
6) 把 `get_charging_profile()` 改成「車種決定功率」，移除 MCS 依 node_type 切換功率  
7) 加入 setup/connect/disconnect；同步修改 evaluate/incremental/forward_propagation  
8) clustering 固定 K=NUM_SLOW；只聚類 normal；Phase 1 建立固定 slow routes  
9) 禁止動態開新車：parallel construction + ALNS repair 全部收掉  
10) 最後才切換成 event-driven（ready_time 事件序列）

---

## 5. 補充：你現有的 “能耗/航程” 檢查要不要改？
你現在 MCS 的能耗檢查寫死 `energy_consumed = distance * 0.5`（kWh/km）：
- 位置：`evaluate_route()`（約 L860-L866）
- 位置：`incremental_insertion_check()`（約 L1046-L1077）

這可以先保留作為 baseline，但建議把 0.5 改成參數：
- `MCS_ENERGY_PER_KM`

UAV 航程檢查目前用 `route.total_distance > MAX_RANGE_KM*2`（假設必須往返 depot）。  
如果 UAV 允許多點串訪，需重新定義（但 baseline 先簡化也 OK）。

---

## 6. 附：你現在最容易踩雷的兩個點（強烈建議先修）

1) **`ChargingSchedulingProblem.__init__` 會直接噴錯**（MCSConfig 未定義）  
2) **finish-time hard TW 會把慢充場景大面積判 infeasible**（應改 start-time TW 或 soft TW）

---

若你希望我再幫你做下一步：  
我可以依照此文件，直接把「要改的函式」逐一列出 pseudo-diff（每個函式應新增/刪除哪些行、改哪些條件判斷），你照著貼進去基本就能跑。
