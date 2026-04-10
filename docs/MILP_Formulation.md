# 完整 MILP 數學模型：HE-EVRPTW-SD（靜態單期版本）

> **問題類別**：Heterogeneous Electric Vehicle Routing Problem with Time Windows and Split Deliveries
> （異質電動車隊路徑問題 + 時間窗 + 拆分配送 + 充電模式選擇）
>
> **建模類型**：3-index Arc-Flow Mixed Integer Linear Programming (MILP)
>
> **適用範圍**：靜態單期最佳化（**不含 Rolling-Horizon、不含隨機到達**），目標規模 $|N| \in [10, 30]$

針對本論文 EV 充電車隊路由問題的完整混合整數線性規劃建模。採用標準三索引弧流（3-index arc-flow）形式，配合所有非標準特徵（UAV Safe-SOC、拆分配送、雙充電模式）。

**用途定位**：作為**小規模 instance 的最佳解 benchmark**，與 ALNS 啟發式進行 optimality gap 對照，以證明啟發式的解品質。對齊既有靜態實驗腳本（`exp_route_comparison.py`、`exp_coverage_vs_demand.py`），但專注於 MILP 可求得最佳解的範圍 $|N| \leq 30$。

---

## 1. 集合與索引（Sets and Indices）

| 符號 | 意義 |
|------|------|
| $N$ | 充電請求節點集合，索引 $i, j$ |
| $N_u \subseteq N$ | 急迫請求子集 |
| $N_n \subseteq N$ | 一般請求子集，$N = N_u \cup N_n$ |
| $0$ | 起始 depot |
| $0'$ | 終點 depot（複本，用於閉合路線） |
| $V = N \cup \{0, 0'\}$ | 全節點集合 |
| $A = \{(i,j) : i,j \in V, i \neq j\}$ | 弧集合 |
| $K$ | 車隊集合 |
| $K_s \subseteq K$ | MCS-Slow 車隊 |
| $K_f \subseteq K$ | MCS-Fast 車隊 |
| $K_u \subseteq K$ | UAV 車隊 |
| $K_m = K_s \cup K_f$ | 全部 MCS（地面車輛） |

---

## 2. 參數（Parameters）

### 2.1 距離與時間
| 符號 | 意義 | 單位 |
|------|------|------|
| $d_{ij}^{m}$ | MCS 曼哈頓距離 | km |
| $d_{ij}^{u}$ | UAV 歐氏距離 | km |
| $v_k$ | 車輛 $k$ 速度 | km/min |
| $t_{ijk} = d_{ij}/v_k$ | 弧 $(i,j)$ 旅行時間 | min |
| $r_i$ | 請求 $i$ 釋放時間（ready） | min |
| $\bar{d}_i$ | 請求 $i$ 截止時間（deadline） | min |
| $T$ | 時間視界長度 | min |

### 2.2 能量
| 符號 | 意義 | 單位 |
|------|------|------|
| $D_i$ | 請求 $i$ 的能量需求 | kWh |
| $\text{soc}_i^0$ | 請求 $i$ 對應 EV 的初始 SOC | 比例 |
| $C^{ev}$ | EV 電池容量 | kWh (60) |
| $E_k^{cap}$ | 車輛 $k$ 電池容量 | kWh |
| $E_k^{0}$ | 車輛 $k$ 初始能量 | kWh |
| $\mu_k$ | 車輛 $k$ 行駛能耗率 | kWh/km |
| $P^{slow}, P^{fast}$ | MCS 慢/快充電功率 | kW |
| $P^{uav}$ | UAV 充電功率 | kW |
| $Q^{uav}$ | UAV 單次最大載能 | kWh (20) |
| $\sigma$ | UAV 目標 SOC 比例 | 0.3 |

### 2.3 目標函數權重
| 符號 | 意義 | 預設值 |
|------|------|--------|
| $\alpha_u$ | 急迫未服務懲罰 | 3500 |
| $\alpha_n$ | 一般未服務懲罰 | — |
| $\beta_u$ | 急迫等待時間單位成本 | 5 |
| $\beta_n$ | 一般等待時間單位成本 | — |
| $\gamma$ | 距離單位成本 | 1 |
| $M$ | 大常數（Big-M） | 充分大 |

---

## 3. 決策變數（Decision Variables）

### 3.1 二元變數

$$
x_{ijk} = \begin{cases} 1, & \text{若車輛 } k \text{ 經過弧 } (i,j) \\ 0, & \text{否則} \end{cases} \quad \forall (i,j) \in A,\, k \in K
$$

$$
y_{ik} = \begin{cases} 1, & \text{若車輛 } k \text{ 服務節點 } i \\ 0, & \text{否則} \end{cases} \quad \forall i \in N,\, k \in K
$$

$$
z_i = \begin{cases} 1, & \text{若節點 } i \text{ 未被服務} \\ 0, & \text{否則} \end{cases} \quad \forall i \in N
$$

$$
\delta_{ik}^{s},\, \delta_{ik}^{f} \in \{0,1\} \quad \text{(MCS 慢/快充電模式)} \quad \forall i \in N,\, k \in K_m
$$

### 3.2 連續變數

| 變數 | 意義 | 範圍 |
|------|------|------|
| $a_{ik}$ | 車輛 $k$ 抵達節點 $i$ 的時間 | $\geq 0$ |
| $b_{ik}$ | 服務開始時間 | $\geq 0$ |
| $w_{ik}$ | 用戶等待時間 | $\geq 0$ |
| $e_{ik}$ | 車輛 $k$ 在節點 $i$ 的剩餘電量 | $[0, E_k^{cap}]$ |
| $q_{ik}$ | 車輛 $k$ 在節點 $i$ 提供的能量 | $\geq 0$ |
| $q_{ik}^{s}, q_{ik}^{f}$ | 慢/快充配送量輔助變數 | $\geq 0$ |
| $\tau_{ik}$ | 服務時間（充電所需） | $\geq 0$ |

---

## 4. 目標函數（Objective Function）

$$
\min Z \;=\; \underbrace{\alpha_u \sum_{i \in N_u} z_i + \alpha_n \sum_{i \in N_n} z_i}_{\text{(I) 未服務懲罰}} + \underbrace{\beta_u \sum_{i \in N_u} \sum_{k \in K} w_{ik} + \beta_n \sum_{i \in N_n} \sum_{k \in K} w_{ik}}_{\text{(II) 應答時間成本}} + \underbrace{\gamma \sum_{(i,j) \in A} \sum_{k \in K} d_{ijk} \cdot x_{ijk}}_{\text{(III) 行駛距離成本}}
$$

其中 $d_{ijk} = d_{ij}^{m}$ 若 $k \in K_m$，否則 $d_{ij}^{u}$。

---

## 5. 約束條件（Constraints）

### 5.1 指派與流量守恆

**(C1) 每個請求必須被服務或標記為未服務**
$$
\sum_{k \in K} y_{ik} + z_i = 1, \quad \forall i \in N
$$

**(C2) 流量守恆（in-degree = out-degree = 服務指示）**
$$
\sum_{j \in V,\, j \neq i} x_{jik} = \sum_{j \in V,\, j \neq i} x_{ijk} = y_{ik}, \quad \forall i \in N,\, k \in K
$$

**(C3) 每車最多離開 depot 一次**
$$
\sum_{j \in N \cup \{0'\}} x_{0jk} \leq 1, \quad \forall k \in K
$$

**(C4) 每車從 depot 出發必返回**
$$
\sum_{j \in N \cup \{0\}} x_{j,0',k} = \sum_{j \in N \cup \{0'\}} x_{0jk}, \quad \forall k \in K
$$

---

### 5.2 時間窗與時間連續性

**(C5) 服務時間定義**

對 MCS：（線性化形式見 §6）
$$
\tau_{ik} = \frac{q_{ik}^{s}}{P^{slow}} + \frac{q_{ik}^{f}}{P^{fast}}, \quad \forall i \in N,\, k \in K_m
$$

對 UAV：
$$
\tau_{ik} = \frac{q_{ik}}{P^{uav}}, \quad \forall i \in N,\, k \in K_u
$$

**(C6) 時間連續性（Big-M 線性化）**
$$
b_{jk} \geq b_{ik} + \tau_{ik} + t_{ijk} - M(1 - x_{ijk}), \quad \forall (i,j) \in A,\, j \neq 0,\, k \in K
$$

**(C7) 服務開始時間定義**
$$
b_{ik} \geq a_{ik}, \quad \forall i \in N,\, k \in K
$$
$$
b_{ik} \geq r_i \cdot y_{ik}, \quad \forall i \in N,\, k \in K
$$

**(C8) 時間窗約束（必須在 deadline 前開始服務）**
$$
b_{ik} + \tau_{ik} \leq \bar{d}_i + M(1 - y_{ik}), \quad \forall i \in N,\, k \in K
$$

**(C9) 等待時間定義**
$$
w_{ik} \geq b_{ik} - r_i - M(1 - y_{ik}), \quad \forall i \in N,\, k \in K
$$
$$
w_{ik} \leq M \cdot y_{ik}
$$

**(C10) Depot 起始時間**
$$
b_{0k} = 0, \quad \forall k \in K
$$

---

### 5.3 能量約束（MCS）

**(C11) 能量初始化**
$$
e_{0k} = E_k^{0}, \quad \forall k \in K
$$

**(C12) 能量沿路線傳播**
$$
e_{jk} \leq e_{ik} - q_{ik} - \mu_k \cdot d_{ijk} + M(1 - x_{ijk}), \quad \forall (i,j) \in A,\, k \in K_m
$$

**(C13) 電量上下界**
$$
0 \leq e_{ik} \leq E_k^{cap}, \quad \forall i \in V,\, k \in K_m
$$

**(C14) 返回 depot 時電量非負**
$$
e_{0'k} \geq 0, \quad \forall k \in K_m
$$

---

### 5.4 充電模式選擇（MCS）

**(C15) 服務時必須選定模式**
$$
\delta_{ik}^{s} + \delta_{ik}^{f} = y_{ik}, \quad \forall i \in N,\, k \in K_m
$$

**(C16) 配送量必須與選定模式一致**
$$
q_{ik}^{s} + q_{ik}^{f} = q_{ik}, \quad \forall i \in N,\, k \in K_m
$$

**(C17) 模式啟動約束（線性化 $q \cdot \delta$）**
$$
q_{ik}^{s} \leq Q^{max} \cdot \delta_{ik}^{s}, \quad q_{ik}^{f} \leq Q^{max} \cdot \delta_{ik}^{f}, \quad \forall i \in N,\, k \in K_m
$$

其中 $Q^{max}$ 為單次最大配送量上界。

---

### 5.5 UAV 特殊約束

**(C18) UAV 載能上限（單趟最大攜帶量）**
$$
q_{ik} \leq Q^{uav} \cdot y_{ik}, \quad \forall i \in N,\, k \in K_u
$$

**(C19) Safe-SOC 上限**

UAV 只將 EV 充到目標 SOC（30%）：
$$
q_{ik} \leq \max\!\Big(0,\; \sigma \cdot C^{ev} - \text{soc}_i^0 \cdot C^{ev}\Big) \cdot y_{ik}, \quad \forall i \in N,\, k \in K_u
$$

**(C20) UAV 飛行能量約束**

對 UAV，能量約束改為：
$$
e_{jk} \leq e_{ik} - \mu_k \cdot d_{ij}^{u} + M(1 - x_{ijk}), \quad \forall (i,j) \in A,\, k \in K_u
$$

（因 UAV 配送的能量來自獨立 payload，不從飛行電池扣除）

---

### 5.6 拆分配送（Split Delivery）

**(C21) 總配送量必須滿足需求（或標為未服務）**
$$
\sum_{k \in K} q_{ik} + D_i \cdot z_i \geq D_i, \quad \forall i \in N
$$

**(C22) 配送量與指派變數連結**
$$
q_{ik} \leq D_i \cdot y_{ik}, \quad \forall i \in N,\, k \in K
$$

**(C23) UAV 服務後 MCS Follow-up（隱含）**

當 UAV 服務不足以滿足需求時，需要 MCS 補足。此由 (C21) 自動處理：若 UAV 只給 $q_{i,uav} < D_i$，則必有某 MCS $k$ 使 $q_{ik} > 0$，否則 $z_i = 1$。

---

### 5.7 變數定義域

$$
x_{ijk}, y_{ik}, z_i, \delta_{ik}^{s}, \delta_{ik}^{f} \in \{0, 1\}
$$
$$
a_{ik}, b_{ik}, w_{ik}, e_{ik}, q_{ik}, q_{ik}^{s}, q_{ik}^{f}, \tau_{ik} \geq 0
$$

---

## 6. 線性化技術摘要

| 非線性項 | 線性化方法 |
|----------|-----------|
| **$q_{ik} \cdot \delta_{ik}$**（雙線性） | 引入 $q_{ik}^{s} = q_{ik} \delta_{ik}^{s}$，配 (C16)(C17) |
| **時間連續性 if-then** | Big-M 法 (C6)(C8)(C9)(C12) |
| **指數級子迴路消除** | 用 MTZ 變數 $u_{ik}$ 或流量守恆隱含消除 |
| **充電功率分段非線性** | 分段線性近似（PWL），引入 SOS2 變數（如需） |

**MTZ 子迴路消除替代方案**（若 (C2)(C3)(C4) 不足）：
$$
u_{ik} - u_{jk} + |N| \cdot x_{ijk} \leq |N| - 1, \quad \forall i,j \in N,\, k \in K
$$

---

## 7. 模型規模分析

對 $|N|$ 個請求、$|K|$ 台車：

| 元素 | 數量 |
|------|------|
| 二元變數 $x_{ijk}$ | $O(\|N\|^2 \|K\|)$ |
| 二元變數 $y_{ik}, \delta_{ik}^s, \delta_{ik}^f$ | $O(\|N\| \|K\|)$ |
| 連續變數 | $O(\|N\| \|K\|)$ |
| 約束數 | $O(\|N\|^2 \|K\|)$ |

### 7.1 目標範圍可解性（$\|N\| \in [10, 30]$，固定車隊 $\|K\| = 6$，3 Slow + 2 Fast + 1 UAV）

| $\|N\|$ | $x_{ijk}$ 變數數 | 約束量級 | Gurobi 預期求解時間 | 預期結果 |
|---------|------------------|---------|---------------------|----------|
| 10 | ~720 | $\sim 600$ | < 10 秒 | ✅ 最佳解 |
| 15 | ~1,620 | $\sim 1,400$ | 數十秒 | ✅ 最佳解 |
| 20 | ~2,880 | $\sim 2,400$ | 1-5 分鐘 | ✅ 最佳解 |
| 25 | ~4,500 | $\sim 3,800$ | 5-30 分鐘 | ✅ 最佳解（多數情況） |
| 30 | ~6,480 | $\sim 5,400$ | 30 分鐘 - 2 小時 | ⚠️ 通常可得最佳，部分 instance 需 time limit |

> **註**：實際時間高度依賴 instance 結構（時間窗鬆緊、需求密度、節點空間分布）。建議統一設定 time limit = 3600 秒，超時則報告 best-feasible + dual bound + gap。

### 7.2 為何上限設為 30

- **30 是 MILP 在合理時間內可達最佳解的上限**：超過 30，Branch-and-Bound tree 急劇膨脹
- **對齊 ALNS 對照需求**：30 個請求已足以展現啟發式優勢，且能與 MILP 最佳解形成有效對照
- **節省計算資源**：避免無意義地長跑無法收斂的大 instance

---

## 8. 求解策略建議

### 8.1 階段性建議
1. **Phase 1**：解 $|N| \in \{10, 15, 20\}$ 小規模 instance → 預期短時間取得最佳解
2. **Phase 2**：擴展到 $|N| \in \{25, 30\}$ → 設 time limit = 3600 秒，記錄最佳解或 best-feasible + gap
3. **Phase 3**：用 ALNS 跑相同 instance（每組 10 次重複），計算 optimality gap

### 8.2 加速技巧
- **Warm Start**：將 ALNS 解作為 MIP start 餵給 Gurobi，可大幅加速 Branch-and-Bound
- **Cuts**：啟用 Gurobi 內建 subset row inequalities、cover cuts
- **Preprocessing**：在建模前移除不可行弧（如 $r_i + s_i + t_{ij} > \bar{d}_j$ 直接固定 $x_{ijk}=0$）
- **變數固定**：若兩節點時間窗不可能在同一路線，固定 $x_{ijk} = 0$
- **對稱性消除**：同型車輛（如 3 台 MCS-Slow）加入車輛索引排序約束，減少對稱解

### 8.3 推薦求解器

| 求解器 | 授權 | 速度 | Python 介面 |
|--------|------|------|-------------|
| **Gurobi** ⭐ | 商業（學術免費） | 最快 | `gurobipy` |
| **CPLEX** | 商業（學術免費） | 快 | `docplex` |
| **HiGHS** | 開源 | 快 | `highspy` |
| **SCIP** | 學術免費 | 中 | `pyscipopt` |
| **COIN-OR CBC** | 開源 | 慢 | `pulp`, `python-mip` |

---

## 9. 完整模型摘要表

| 項目 | 內容 |
|------|------|
| **問題類別** | HE-EVRPTW-SD with Charging Mode Selection |
| **建模類型** | 3-index Arc-Flow MILP（靜態單期） |
| **目標規模** | $\|N\| \in [10, 30]$, $\|K\| = 6$（3 Slow + 2 Fast + 1 UAV） |
| **目標函數項** | 3 項（未服務、等待、距離） |
| **約束類別** | 7 類，共 23 條（不含定義域） |
| **二元變數規模** | $O(\|N\|^2 \|K\|)$ |
| **線性化技巧** | Big-M、雙線性分解、MTZ |
| **複雜度** | NP-hard（VRPTW 已是 NP-hard） |
| **預期 time limit** | 3600 秒/instance |

---

## 10. ALNS vs MILP 對照實驗建議

為證明 ALNS 啟發式品質（IEEE Transactions 審稿人常見要求）：

### 10.1 實驗設計

**固定條件**：車隊 $|K| = 6$（3 MCS-Slow + 2 MCS-Fast + 1 UAV），對齊 `exp_route_comparison.py` 與 `exp_coverage_vs_demand.py`。

| Instance Size | $\|N\|$ | 重複次數（seed） | MILP time limit |
|---------------|---------|------------------|-----------------|
| XS | 10 | 10 | 600 秒 |
| S | 15 | 10 | 1200 秒 |
| M | 20 | 10 | 1800 秒 |
| L | 25 | 10 | 3600 秒 |
| XL | 30 | 10 | 3600 秒 |

**總計**：5 個規模 × 10 個 seed = 50 個 instances

### 10.2 評估指標

$$
\text{Optimality Gap}\% = \frac{Z_{\text{ALNS}} - Z_{\text{MILP}}^{*}}{Z_{\text{MILP}}^{*}} \times 100\%
$$

$$
\text{Time Ratio} = \frac{T_{\text{MILP}}}{T_{\text{ALNS}}}
$$

若 MILP 在 time limit 內未收斂，使用 best-feasible 與 dual bound 計算雙界 gap：

$$
\text{Dual Gap}\% = \frac{Z_{\text{best}} - Z_{\text{LB}}^{\text{dual}}}{Z_{\text{best}}} \times 100\%
$$

### 10.3 期望結果
- $|N| \leq 20$：MILP 取得最佳解，ALNS gap 預期 < 5%
- $|N| \in \{25, 30\}$：MILP 多數可達最佳，少數需 best-feasible，ALNS gap 預期 < 8%
- ALNS 時間遠小於 MILP（通常 100x 以上）
- 證明 ALNS 在小規模上**接近最佳**，因此外推至大規模（$|N| > 100$）的結果可信

### 10.4 對照表格範例（論文用）

| $\|N\|$ | $Z_{\text{MILP}}^{*}$ | $T_{\text{MILP}}$ (s) | $\bar{Z}_{\text{ALNS}}$ | $\bar{T}_{\text{ALNS}}$ (s) | Gap (%) | Speedup |
|---------|----------------------|----------------------|------------------------|----------------------------|---------|---------|
| 10 | ... | ... | ... | ... | ... | ... |
| 15 | ... | ... | ... | ... | ... | ... |
| 20 | ... | ... | ... | ... | ... | ... |
| 25 | ... | ... | ... | ... | ... | ... |
| 30 | ... | ... | ... | ... | ... | ... |
