# 問題定義及環境假設

本節針對本研究所探討之異質行動電動載具充電派遣問題進行正式定義。首先描述服務環境與網路節點分類，接著定義異質充電車隊與距離時間函數，最後闡述最佳化目標及相關約束條件。本問題屬於具時間窗之容量限制車輛路徑問題（Capacitated Vehicle Routing Problem with Time Windows, CVRPTW）的延伸形式，進一步納入具充電能力之無人飛行載具及異質地面充電車輛。

---

## A. 異質性電動車充電請求

考慮一家事業公司，負責接收電動車車主提交的隨機請求 $(x_i, q_i, l_i)$，其中 $x_i$ 表示電動車位置，$q_i$ 表示所需的能量單位（kWh），$l_i$ 表示截止時間，即車主希望在此時間前完成充電。

一組電動車集合 $E = \{1, 2, \ldots, n\}$，這些電動車在前一個時間區間 $\Delta T$ 內提出了充電請求，分布於 $n$ 個不同位置。假設 $x_i$ 與 $x_j$ 分別為第 $i$ 輛與第 $j$ 輛電動車的位置，其中 $i, j \in E$ 且 $i \neq j$。

充電請求依**優先等級**分為兩類子集合：

- **緊急需求集合** $E_u \subseteq E$：時間窗較短，需於有限時間內完成服務
- **一般需求集合** $E_n \subseteq E$：時間窗較寬裕，$E_u \cup E_n = E$，$E_u \cap E_n = \emptyset$

每個請求之屬性如下：

| 符號 | 定義 |
|------|------|
| $(x_i, y_i)$ | 電動車 $i$ 之座標位置 |
| $q_i$ | 電動車 $i$ 之充電需求量 (kWh)，$q_i \sim \mathcal{TN}(\mu_q, \sigma_q^2, q^{\min}, q^{\max})$ |
| $t_i$ | 電動車 $i$ 之請求到達時間（ready time） |
| $l_i$ | 電動車 $i$ 之服務截止時間（due date），$l_i = t_i + w_i$ |
| $w_i$ | 電動車 $i$ 之時間窗寬度 |
| $s_i^{0}$ | 電動車 $i$ 之初始荷電狀態（SOC） |
| $q_i^{\text{orig}}$ | 電動車 $i$ 之原始完整需求量 |
| $q_i^{\text{res}}$ | 電動車 $i$ 經 UAV 服務後之剩餘需求量 |

**需求類型參數**：

| 參數 | Urgent ($E_u$) | Normal ($E_n$) |
|------|--------|--------|
| $q^{\min}$ (kWh) | 2.0 | 2.0 |
| $q^{\max}$ (kWh) | 20.0 | 20.0 |
| $\mu_q$ (kWh) | 6.9 | 6.9 |
| $\sigma_q$ (kWh) | 4.9 | 4.9 |
| $w^{\min}$ (min) | 30 | 480 |
| $w^{\max}$ (min) | 60 | 600 |
| $s^{0}$ 範圍 | $[0.05, 0.20]$ | 由需求推算 |

---

## B. 異質行動充電車隊

假設 $S^U = \{1, 2, \ldots, m^U\}$、$S^F = \{1, 2, \ldots, m^F\}$、$S^C = \{1, 2, \ldots, m^C\}$ 分別為 UAV、快速充電車與慢速充電車的集合，數量分別為 $m^U$、$m^F$、$m^C$ 台。令 $S = S^U \cup S^F \cup S^C$ 為所有充電資源的集合，共 $m = m^U + m^F + m^C$ 個資源，需透過 $m$ 條不同路線進行路徑規劃，以服務 $n$ 輛電動車。

$G_k^U = \{g_k^U(1), \ldots, g_k^U(n_k^U)\}$ 表示第 $k$ 架 UAV 服務序列（$\forall k \in S^U$），其中 $g_k^U(j)$ 代表第 $k$ 架 UAV 所服務的第 $j$ 輛電動車，$n_k^U$ 為其服務的 EV 總數。$G_k^F = \{g_k^F(1), \ldots, g_k^F(n_k^F)\}$ 與 $G_k^C = \{g_k^C(1), \ldots, g_k^C(n_k^C)\}$ 分別表示第 $k$ 輛快速充電車（$\forall k \in S^F$）與第 $k$ 輛慢速充電車（$\forall k \in S^C$）的服務序列，$n_k^F$、$n_k^C$ 分別為其服務數量。

假設存在一個位於座標 $x_0$ 的中央充電設施（depot）。所有充電資源完成服務後須返回中央充電設施進行補充，因此：

$$g_k^U(n_k^U + 1) = 0, \quad \forall k \in S^U$$

$$g_k^F(n_k^F + 1) = 0, \quad \forall k \in S^F$$

$$g_k^C(n_k^C + 1) = 0, \quad \forall k \in S^C$$

由於三類資源在移動速度、充電容量與充電速率上具有差異，本研究將車隊視為**異質性車隊**。三類充電資源之參數如下：

| 符號 | 定義 | MCS-Slow ($S^C$) | MCS-Fast ($S^F$) | UAV ($S^U$) |
|------|------|----------|----------|-----|
| $\rho^s$ | 充電速率 (kW) | $\rho^C$ | $\rho^F$ | $\rho^U$ |
| $Q$ | 充電模組容量 (kWh) | $Q^G$ | $Q^G$ | $Q^U$ |
| $\nu$ | 移動速度 (km/min) | $\nu^G$ | $\nu^G$ | $\nu^U$ |
| $\epsilon$ | 行駛能耗 (kWh/km) | $\epsilon^G$ | $\epsilon^G$ | $\epsilon^U$ |
| 距離度量 | — | Manhattan | Manhattan | Euclidean |

**共用服務開銷參數**：所有充電資源於服務電動車時，皆需經歷連線與斷線程序。

| 符號 | 定義 |
|------|------|
| $\tau^{\text{conn}}$ | 連線時間：將充電設備接上電動車之操作時間 (min) |
| $\tau^{\text{disc}}$ | 斷線時間：充電完成後拔除充電設備之操作時間 (min) |

**UAV 額外開銷參數**：UAV 每趟飛行任務另需起飛與降落之作業時間。

| 符號 | 定義 |
|------|------|
| $\tau^{\text{fly}}$ | 起降作業時間：UAV 每趟飛行任務之起飛與降落總開銷 (min) |

快速充電車與慢速充電車具有相同的移動速度 $\nu^G$ 與電池容量 $Q^G$，兩者的差異僅在於充電速率，滿足 $\rho^F > \rho^C$。UAV 的移動速度高於地面車 $\nu^U > \nu^G$，但其攜帶之充電模組容量遠小於地面車 $Q^U \ll Q^G$。三類資源之充電速率滿足 $\rho^F > \rho^U > \rho^C$。

**UAV 特有參數**：

| 符號 | 定義 |
|------|------|
| $Q^U$ | UAV 單趟最大可交付能量 (kWh) |
| $s^{\text{target}}$ | UAV 目標充電 SOC |
| $C_{\text{EV}}$ | EV 電池容量 (kWh)，用於 SOC 換算 |
| $R_{\text{max}}$ | UAV 最大航程 (km) |
| $T_{\text{endurance}}$ | UAV 續航時間 (min) |

**後備車輛集合**：令 $S^R \subseteq S$ 為後備（reserve）車輛集合，初始為非啟用狀態，可依需求動態啟用。

---

## C. 距離與時間函數

地面充電車適用**曼哈頓距離**：

$$d^{Man}(i, j) = |x_i - x_j| + |y_i - y_j|$$

無人飛行載具適用**歐幾里得距離**：

$$d^{Euc}(i, j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

**車輛行駛距離**依車輛類型選擇適用之距離度量：

$$
d_k(i, j) =
\begin{cases}
d^{Man}(i, j), & \text{if } k \in S^F \cup S^C \\
d^{Euc}(i, j), & \text{if } k \in S^U
\end{cases}
$$

**行駛時間**：

$$
\theta_k(i, j) =
\begin{cases}
\dfrac{d^{Man}(i, j)}{\nu^G}, & \text{if } k \in S^F \cup S^C \\[8pt]
\dfrac{d^{Euc}(i, j)}{\nu^U}, & \text{if } k \in S^U
\end{cases}
$$

**充電時間**：充電資源 $k$ 對電動車 $i$ 的純充電時間 $s_i^k$，由該車充電量 $q_i$ 與所屬類型之充電速率 $\rho^s$ 決定：

$$s_i^k = \frac{q_i}{\rho^s}, \quad \forall k \in S^s,\ s \in \{U, F, C\}$$

其中 $\rho^s$ 依車輛類型取值：$k \in S^F$ 時 $\rho^s = \rho^F$，$k \in S^C$ 時 $\rho^s = \rho^C$，$k \in S^U$ 時 $\rho^s = \rho^U$。

**駐點服務時間** $f^{\text{svc}}(i, k)$：充電資源 $k$ 到達電動車 $i$ 位置後，至完成服務離開所需之時間。

MCS 需經歷連線、充電、斷線三個階段：

$$f_G^{\text{svc}}(i, k) = \tau^{\text{conn}} + s_i^k + \tau^{\text{disc}}, \quad \forall k \in S^F \cup S^C$$

UAV 需經歷起降作業、連線、充電、斷線四個階段：

$$f_U^{\text{svc}}(i, k) = \tau^{\text{fly}} + \tau^{\text{conn}} + s_i^k + \tau^{\text{disc}}, \quad \forall k \in S^U$$

統一表達：

$$
f^{\text{svc}}(i, k) =
\begin{cases}
\tau^{\text{conn}} + s_i^k + \tau^{\text{disc}}, & \text{if } k \in S^F \cup S^C \\[6pt]
\tau^{\text{fly}} + \tau^{\text{conn}} + s_i^k + \tau^{\text{disc}}, & \text{if } k \in S^U
\end{cases}
$$

**完整服務週期** $f^{\text{total}}(i, k)$：充電資源 $k$ 從上一個服務節點（或 depot）出發，行駛至電動車 $i$ 並完成服務離開之總耗時。包含行駛時間與駐點服務時間：

$$f^{\text{total}}(i, k) = \theta_k(\text{prev}, i) + f^{\text{svc}}(i, k)$$

其中 $\theta_k(\text{prev}, i)$ 為從前一節點至節點 $i$ 之純行駛時間。展開各類型：

$$
f^{\text{total}}(i, k) =
\begin{cases}
\dfrac{d^{Man}(\text{prev}, i)}{\nu^G} + \tau^{\text{conn}} + s_i^k + \tau^{\text{disc}}, & \text{if } k \in S^F \cup S^C \\[10pt]
\dfrac{d^{Euc}(\text{prev}, i)}{\nu^U} + \tau^{\text{fly}} + \tau^{\text{conn}} + s_i^k + \tau^{\text{disc}}, & \text{if } k \in S^U
\end{cases}
$$

---

## D. 決策變數

| 變數 | 定義 |
|------|------|
| $y_i^k \in \{0, 1\}$ | 電動車 $i$ 是否被分配至充電資源 $k$，$\forall i \in E,\ k \in S$ |
| $z_i \in \{0, 1\}$ | 電動車 $i$ 是否未被服務（missed），$\forall i \in E$ |
| $a_k \in \{0, 1\}$ | 後備車輛 $k$ 是否被啟用，$\forall k \in S^R$ |
| $\tau_i^k$ | 充電資源 $k$ 到達電動車 $i$ 之時間 |
| $B_i^k$ | 電動車 $i$ 之服務開始時間（由資源 $k$ 服務） |
| $D_i^k$ | 充電資源 $k$ 完成電動車 $i$ 服務後之離開時間 |

---

## E. 約束條件

### E.1 服務指派約束

$$\sum_{k \in S} y_i^k + z_i = 1, \quad \forall i \in E \tag{C1}$$

每個充電需求必須且僅能被分配至一個充電資源，或者被標記為未服務（missed service）。

### E.2 時間相關輔助定義

為描述截止時間約束，先定義服務開始時間與離開時間。

**服務開始時間**：充電資源 $k$ 到達電動車 $i$ 後，需經服務前置作業方可開始充電，且不得早於請求到達時間 $t_i$：

$$B_i^k = \max\left(\tau_i^k + \tau_k^{\text{oh,pre}},\ t_i\right), \quad \forall i \in E,\ k \in S \tag{A1}$$

其中服務前置開銷 $\tau_k^{\text{oh,pre}}$ 依資源類型定義如下：

$$
\tau_k^{\text{oh,pre}} =
\begin{cases}
\tau^{\text{conn}}, & \text{if } k \in S^F \cup S^C \\
\tau^{\text{fly}} + \tau^{\text{conn}}, & \text{if } k \in S^U
\end{cases}
$$

**離開時間**：充電完成後經斷線程序方可離開，三類資源共用相同之後置開銷 $\tau^{\text{disc}}$：

$$D_i^k = B_i^k + s_i^k + \tau^{\text{disc}}, \quad \forall i \in E,\ k \in S \tag{A2}$$

**到達時間傳遞**：路徑 $\text{route}(k)$ 上，若節點 $j$ 緊接於節點 $i$ 之後被服務，則資源 $k$ 到達 $j$ 之時間不得早於其離開 $i$ 之時間加上 $i$ 至 $j$ 之行駛時間：

$$\tau_j^k \geq D_i^k + \theta_k(i, j), \quad \forall (i, j) \in \text{route}(k),\ i \prec j \tag{A3}$$

### E.3 截止時間約束

充電資源 $k$ 必須於截止時間 $l_i$ 前完成對電動車 $i$ 之服務並離開，此約束僅於 $i$ 被指派至 $k$ 時生效：

$$D_i^k \leq l_i + M(1 - y_i^k), \quad \forall i \in E,\ k \in S \tag{C2}$$

其中 $M$ 為一充分大的常數。

### E.4 地面充電車容量約束

地面充電車具有相同電池容量 $Q^G$，單趟總配送能量不得超過此上限：

$$\sum_{i \in E} q_i \cdot y_i^k \leq Q^G, \quad \forall k \in S^F \cup S^C \tag{C3}$$

### E.5 UAV 可交付能量約束

UAV 電池容量為 $Q^U$（$Q^U \ll Q^G$），其中 $\hat{q}_i = \min(q_i,\ Q^U)$ 為 UAV 對需求 $i$ 之實際交付電量：

$$\sum_{i \in E} \hat{q}_i \cdot y_i^k \leq Q^U, \quad \forall k \in S^U \tag{C4}$$

### E.6 Depot 可達性約束（返回電量）

所有充電資源完成服務後，須保留足夠電量返回中央充電設施。令 $\text{last}(k)$ 表示資源 $k$ 路徑上最後服務之電動車。

**地面充電車**：

$$E_k^{\text{rem}} \geq \epsilon^G \cdot d^{Man}(\text{last}(k),\ 0), \quad \forall k \in S^F \cup S^C \tag{C5a}$$

其中 $E_k^{\text{rem}}$ 為完成所有服務後之剩餘能量：

$$E_k^{\text{rem}} = Q^G - \sum_{(i,j) \in \text{route}(k)} \epsilon^G \cdot d^{Man}(i,j) - \sum_{i \in \text{route}(k)} q_i$$

**UAV**：UAV 每趟僅服務一個需求即返回 depot，故單趟任務之總能耗（來回飛行＋充電交付）不得超過其電池容量：

$$\epsilon^U \cdot d^{Euc}(0,\ i) + \hat{q}_i + \epsilon^U \cdot d^{Euc}(i,\ 0) \leq Q^U, \quad \forall i \in E,\ k \in S^U,\ y_i^k = 1 \tag{C5b}$$

即 UAV 必須保有足夠電量完成 depot → EV $i$ → depot 之完整來回任務。

### E.7 UAV 重複服務禁止約束

定義參數 $\bar{u}_i \in \{0, 1\}$，若電動車 $i$ 於前期時間區間已被 UAV 服務過則 $\bar{u}_i = 1$，否則為 $0$。已被 UAV 服務之需求，其後續剩餘需求僅能由地面充電車處理：

$$y_i^k \leq 1 - \bar{u}_i, \quad \forall i \in E,\ k \in S^U \tag{C6}$$

### E.8 後備車輛啟用約束

$$y_i^k \leq a_k, \quad \forall i \in E,\ k \in S^R \tag{C7}$$

後備車輛僅在被啟用（$a_k = 1$）後方可服務需求。

### E.9 變數定義域

$$y_i^k \in \{0, 1\},\quad z_i \in \{0, 1\},\quad a_k \in \{0, 1\}$$

$$\tau_i^k, B_i^k, D_i^k \geq 0$$

---

## F. 最佳化目標

最佳化目標為最小化以下加權複合成本函數：

$$\min\ Z = \omega_1 \cdot N_{us} + \omega_2 \cdot P_{tw} + \omega_3 \cdot C_{tr}$$

其中：

**未服務需求懲罰**（依需求類型加權）：

$$N_{us} = \alpha_u \sum_{i \in E_u} z_i + \alpha_n \sum_{i \in E_n} z_i$$

其中 $\alpha_u$ 為緊急需求未服務懲罰權重，$\alpha_n$ 為一般需求未服務懲罰權重，$\alpha_u > \alpha_n$。

**時間窗違反總懲罰**：

$$P_{tw} = \sum_{k \in S} T_k$$

其中 $T_k$ 為充電資源 $k$ 之路徑總時間（含行駛、等待、充電及返回 depot）：

$$T_k = \sum_{(i,j) \in \text{route}(k)} \left[\theta_k(i, j) + f^{\text{svc}}(j, k) \cdot \mathbf{1}[j \neq 0]\right]$$

**總行駛成本**（與距離成正比）：

$$C_{tr} = \sum_{k \in S} D_k^{\text{total}}$$

$$D_k^{\text{total}} = \sum_{(i,j) \in \text{route}(k)} d_k(i, j)$$

三項加權係數設定為 $\omega_1 \gg \omega_2 \gg \omega_3$，以確保目標函數依序以下列**優先序**考量：

1. **最小化未服務需求數**（最高優先）— 系統應優先確保每筆充電請求均能被服務
2. **最小化路徑總時間**— 其次考量是否於指定時間窗內完成
3. **最小化行駛距離成本**（最低優先）— 最後追求路線效率的最小化

此設計反映實際營運需求：服務覆蓋率 > 時效性 > 路線效率。

**覆蓋率指標**：

$$Coverage = \frac{Total\ Served}{Total\ Generated} = \frac{\sum_{i \in E}(1 - z_i)}{|E|}$$

---

## G. UAV 部分充電機制

UAV 採用「**安全 SOC 充電**」策略，而非完全滿足原始需求。由於 $Q^U \ll Q^G$，UAV 僅提供緊急安全充電，將 EV 充至目標 SOC 後，剩餘需求重新排入地面充電車佇列。

**UAV 實際交付電量**：

$$\hat{q}_i = \min\left(q_i,\ Q^U\right) \tag{P1}$$

**服務後 EV 之 SOC 更新**：

$$s_i^{\text{after}} = s_i^{0} + \frac{\hat{q}_i}{C_{\text{EV}}} \tag{P2}$$

**剩餘需求計算**：

$$q_i^{\text{res}} = q_i^{\text{orig}} - \hat{q}_i \tag{P3}$$

**後續需求重排規則**（UAV Re-queue）：若 $q_i^{\text{res}} > q^{\text{threshold}}$，系統生成一後續 normal 類型需求節點 $i'$，具有以下屬性：

| 屬性 | 值 |
|------|-----|
| 座標 | $(x_{i'}, y_{i'}) = (x_i, y_i)$ |
| 需求 | $q_{i'} = q_i^{\text{res}}$ |
| 到達時間 | $t_{i'} = t_k^{\text{end}}$（下一決策時點） |
| 截止時間 | $l_{i'} = t_{i'} + w_{i'}$，$w_{i'} \sim \mathcal{U}(w_n^{\min}, w_n^{\max})$ |
| 類型 | normal |
| UAV 服務標記 | $\text{uav\_served}(i') = \text{true}$（禁止 UAV 重複服務） |

---

## H. 滾動規劃派遣機制

### H.1 時間結構

排程時域 $[0, T]$ 劃分為若干等長時間槽，每槽長度為 $\Delta T$。整個時域共 $K = T / \Delta T$ 個時間槽：

$$\mathcal{I}_k = [(k-1)\Delta T,\ k\Delta T), \quad k = 1, 2, \ldots, K$$

每個時間槽結束時（即派遣時間點 $t_k = k \cdot \Delta T$），中央排程器收集所有待處理需求，包含新進需求及前一槽未完成之遺留需求，執行一次完整的路線重規劃。

### H.2 需求到達模型

在時間槽 $k$ 內到達之需求數 $N_k$ 依據 Non-Homogeneous Poisson Process 生成：

$$N_k \sim \text{Poisson}(\mu_k), \quad \mu_k = \bar{\lambda} \cdot r(t_k^s) \cdot \Delta T$$

其中 $\bar{\lambda}$ 為平均到達率，$r(t)$ 為時變到達率輪廓函數。每個需求之到達時間 $t_i \sim \mathcal{U}(t_k^s, t_k^e)$，座標 $(x_i, y_i) \sim \mathcal{U}([0, L]^2)$。

### H.3 車輛釋放狀態

在調度時點 $t_k$，每台充電資源 $k$ 之狀態以三元組表達：

$$\mathcal{S}_k^{(t)} = \left(\text{pos}_k,\ \tau_k^{\text{avail}},\ E_k^{\text{rem}}\right)$$

- $\text{pos}_k$：充電資源完成所有已承諾任務後之預期位置（release node）
- $\tau_k^{\text{avail}}$：最早可再接受新任務之時間（release time）
- $E_k^{\text{rem}}$：預期剩餘能量（release energy）

### H.4 凍結前綴

充電資源在調度時點已開始執行或正在前往之任務稱為「**已承諾節點**」（committed nodes），構成**凍結前綴**（frozen prefix）。這些任務不可被重新規劃。新決策僅決定**路徑後綴**（route suffix），每條路線的起點為車輛當前的真實狀態——當前位置、可出發時間、剩餘電量——而非每輪從中央充電設施重新出發。

### H.5 活動請求池

在時間槽 $k$ 之調度時點，活動請求池由以下組成：

$$\mathcal{P}_k = E_k^{\text{new}} \cup \mathcal{B}_k^{\text{active}}$$

其中 $E_k^{\text{new}}$ 為本期新到達之需求，$\mathcal{B}_k^{\text{active}}$ 為上期未服務且尚未逾期之遺留需求（backlog）：

$$\mathcal{B}_k^{\text{active}} = \{i \in \mathcal{B}_{k-1} : l_i > t_k\}$$

時間窗於服務前即已超時之需求，則記錄為未服務並計入目標函數之懲罰項：

$$\{i \in \mathcal{B}_{k-1} : l_i \leq t_k\} \rightarrow z_i = 1$$

於本時間槽內無法完成服務之需求（例如所有車輛電池容量已耗盡），將被累積並於下一槽重新排定優先序。

### H.6 每時間槽決策流程

```
Algorithm: Rolling-Horizon Batch Dispatching
─────────────────────────────────────────────
Input:  T, ΔT, fleet S, arrival process parameters
Output: Scheduling decisions and service records for all slots

For k = 1, 2, ..., K:
  Step 1 — 處理待辦清單：
           逾期者 (l_i ≤ t_k) 標記 missed
           有效者加入 active pool P_k

  Step 2 — 凍結已執行請求：
           已完成者標記「已服務」
           執行中者保留為 committed，不重複派遣

  Step 3 — 合併請求並重新規劃：
           P_k = E_k^new ∪ B_k^active
           對 P_k 求解充電排程問題

  Step 4 — 更新待辦清單供下輪使用：
           未指派且 l_i > t_{k+1} 者加入新待辦清單
           已指派但未開始服務者退回待辦清單重排
           其餘標記為 missed
```

### H.7 待辦清單請求來源

待辦清單的請求來源有兩類：

1. 上輪執行貪婪插入後仍落入未指派清單、且截止時間尚未到期的請求
2. 上輪雖已被指派至某條路線，但在該執行窗口內車輛尚未實際開始服務的請求——這類請求在下一輪會退回待辦清單重新排入規劃

---

## I. 求解方法

### I.1 貪婪插入建構啟發式

**請求排序**：所有本輪有效請求池中的請求依最早截止時間 $l_i$ 優先排序，優先處理截止時間最早的請求，以降低逾期遺失率。

**三層枚舉與插入**：對排序後的每筆請求，系統執行以下流程：

1. 枚舉所有充電資源路線（候選路線）
2. 枚舉每條路線中的所有插入位置
3. 對每個候選位置進行可行性與成本增量評估
4. 從所有可行位置中選出成本增量最小者
5. 若找到可行位置，將請求插入該路線的最佳位置並標記為已指派
6. 若無任何可行位置，將請求加入未指派清單

**成本增量**定義為：

$$\Delta C = \text{插入後路線總完成時間} - \text{插入前路線總完成時間}$$

系統選擇所有可行插入方案中成本增量最小者，即對路線整體影響最小的插入方式。

**充電資源啟用優先序**：

| 優先序 | 充電資源類型 | 說明 |
|--------|---------|------|
| 1 | $S^C$ (deployed) | 優先使用已在途之慢速充電車 |
| 2 | $S^C$ (idle) | 使用閒置之慢速充電車 |
| 3 | $S^F$ (deployed) | 快速充電車（已在途） |
| 4 | $S^F$ (idle) | 快速充電車（閒置） |
| 5 | $S^U$ | 僅當地面充電車均不可行時，作為 fallback |
| 6 | $S^R$ | 啟用後備車輛 |

**可行性三道硬性約束**：系統對候選路線逐站模擬，依序進行三項約束檢查，任一不通過即提前回傳不可行：

1. **截止時間約束**：$D_i^k \leq l_i$，完整服務流程（抵達、設定、連線、充電、斷線）須全部在截止時間前完成
2. **服務電量約束**：扣除行駛耗電與充電輸出後，車輛剩餘電量 $\geq 0$
3. **返回電量約束**：路線中所有請求服務完畢後，車輛仍需保留足夠電量返回中央充電設施

### I.2 ALNS 改善啟發式

本問題採用**自適應大鄰域搜索（Adaptive Large Neighborhood Search, ALNS）** 配合**模擬退火（Simulated Annealing, SA）** 接受準則。

**破壞運算子**（Destroy Operators）：

- **Random Removal**：隨機移除 $\lfloor \delta \cdot |\text{served}| \rfloor$ 個節點，$\delta \in [\delta_{\min}, \delta_{\max}]$
- **Worst Removal**：移除使用者等待時間最長之節點

**修復運算子**（Repair Operators）：

- **Greedy Insertion**：依 $l_i$ 順序，逐一插入邊際路徑時間增量最小之位置
- **Regret-2 Insertion**：選擇次佳與最佳插入成本差異（regret value）最大之節點優先插入

**SA 接受準則**：

$$
P(\text{accept}) =
\begin{cases}
1, & \text{if } \Delta Z < 0 \\
\exp(-\Delta Z / T_{\text{SA}}), & \text{otherwise}
\end{cases}
$$

$$T_{\text{SA}}^{(t+1)} = \beta \cdot T_{\text{SA}}^{(t)}, \quad T_{\text{SA}}^{(0)} = T_0$$

**自適應權重更新**：每 $\sigma_{\text{seg}}$ 次迭代更新運算子權重：

$$w_j^{(s+1)} = (1 - \lambda_r) \cdot w_j^{(s)} + \lambda_r \cdot \frac{\pi_j^{(s)}}{\theta_j^{(s)}}$$

其中 $\pi_j^{(s)}$ 為運算子 $j$ 在區段 $s$ 之累積分數，$\theta_j^{(s)}$ 為使用次數。分數依改善程度分三級：$\sigma_1$（新全域最佳）、$\sigma_2$（改善當前解）、$\sigma_3$（SA 接受之劣解）。
