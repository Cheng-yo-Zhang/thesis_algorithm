# Parallel Insertion Construction — 完整 Trace

> Source: `main_nocluster.py` / `instance_10c_random_s42.csv` (10 customers)

## 參數速查

| 參數 | 值 |
|---|---|
| MCS Speed | 0.7200 km/min (12 m/s) |
| MCS Capacity | 270.0 kWh |
| MCS Fast Power | 250.0 kW |
| MCS Slow Power | 11.0 kW |
| UAV Speed | 1.0 km/min |
| UAV Deliverable Energy | 20.0 kWh |
| UAV Charge Power | 50.0 kW |
| UAV Overhead | 3.0 min |
| Depot due_date | 1440.0 min |

## 插入順序 (Urgent 優先, 再按 due_date 排)

| 順序 | Node | Type | (x,y) | Demand | Ready | Due | Service |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 11 | urgent | (29.0,37.0) | 16.0 | 744.0 | 807.0 | 9.0 |
| 2 | 9 | urgent | (21.0,52.0) | 18.0 | 1073.0 | 1136.0 | 10.0 |
| 3 | 7 | normal | (87.0,99.0) | 11.0 | 131.0 | 618.0 | 68.0 |
| 4 | 10 | normal | (1.0,87.0) | 43.0 | 202.0 | 687.0 | 252.0 |
| 5 | 8 | normal | (23.0,2.0) | 16.0 | 167.0 | 724.0 | 97.0 |
| 6 | 2 | normal | (51.0,92.0) | 19.0 | 413.0 | 901.0 | 114.0 |
| 7 | 3 | normal | (14.0,71.0) | 17.0 | 406.0 | 938.0 | 103.0 |
| 8 | 4 | normal | (60.0,20.0) | 11.0 | 418.0 | 989.0 | 68.0 |
| 9 | 5 | normal | (82.0,86.0) | 50.0 | 531.0 | 1070.0 | 292.0 |
| 10 | 6 | normal | (74.0,74.0) | 13.0 | 563.0 | 1086.0 | 80.0 |

---
## Step 1: 插入 Node 11

- **Type**: urgent  &emsp; **Coord**: (29.0, 37.0)  &emsp; **Demand**: 16.0 kWh
- **Ready**: 744.0  &emsp; **Due**: 807.0  &emsp; **Service**: 9.0 min

### 當前路徑狀態

*(尚無任何路徑)*

### 候選位置評估

*(尚無現有路徑可嘗試)*

### 新開路徑比較

| Vehicle | Feasible | Wait (min) | Depart | Travel | Arrival | Mode | Charge Time |
|:---:|:---:|---:|---:|---:|---:|:---:|---:|
| New MCS | OK | 47.22 | 744 | 47.22 | 791.22 | FAST | 3.84 |
| New UAV | OK | 27.70 | 744 | 27.70 | 771.70 | FAST | 19.20 |

### 決策

**新開 UAV-0** (UAV wait 27.70 <= MCS wait 47.22)

### 插入後路徑快照

- **UAV-0**: D -> 11 -> D
  - N11: arr=771.70, dep=790.90, wait=27.70

---
## Step 2: 插入 Node 9

- **Type**: urgent  &emsp; **Coord**: (21.0, 52.0)  &emsp; **Demand**: 18.0 kWh
- **Ready**: 1073.0  &emsp; **Due**: 1136.0  &emsp; **Service**: 10.0 min

### 當前路徑狀態

- **UAV-0**: D -> 11 -> D &emsp; demand=16, dist=49.40, time=818.60

### 候選位置評估

| Route | Pos | Feasible | delta_cost | time_shift | Constraint / Reason |
|:---:|:---:|:---:|---:|---:|:---|
| UAV-0 | 0 | FAIL | inf | --- | CAPACITY (route_demand=16.00, node_demand=18.00, capacity=20.00) |
| UAV-0 | 1 | FAIL | inf | --- | CAPACITY (route_demand=16.00, node_demand=18.00, capacity=20.00) |

### 新開路徑比較

| Vehicle | Feasible | Wait (min) | Depart | Travel | Arrival | Mode | Charge Time |
|:---:|:---:|---:|---:|---:|---:|:---:|---:|
| New MCS | OK | 43.06 | 1073 | 43.06 | 1116.06 | FAST | 4.32 |
| New UAV | OK | 32.07 | 1073 | 32.07 | 1105.07 | FAST | 21.60 |

### 決策

**新開 UAV-1** (UAV wait 32.07 <= MCS wait 43.06)

### 插入後路徑快照

- **UAV-0**: D -> 11 -> D
  - N11: arr=771.70, dep=790.90, wait=27.70
- **UAV-1**: D -> 9 -> D
  - N9: arr=1105.07, dep=1126.67, wait=32.07

---
## Step 3: 插入 Node 7

- **Type**: normal  &emsp; **Coord**: (87.0, 99.0)  &emsp; **Demand**: 11.0 kWh
- **Ready**: 131.0  &emsp; **Due**: 618.0  &emsp; **Service**: 68.0 min

### 當前路徑狀態

- **UAV-0**: D -> 11 -> D &emsp; demand=16, dist=49.40, time=818.60
- **UAV-1**: D -> 9 -> D &emsp; demand=18, dist=58.14, time=1158.74

### 候選位置評估

*(尚無現有路徑可嘗試)*

### 新開路徑比較

| Vehicle | Feasible | Wait (min) | Depart | Travel | Arrival | Mode | Charge Time |
|:---:|:---:|---:|---:|---:|---:|:---:|---:|
| New MCS | OK | 119.44 | 131 | 119.44 | 250.44 | SLOW | 60.00 |

### 決策

**新開 MCS-0**

### 插入後路徑快照

- **MCS-0**: D -> 7 -> D
  - N7: arr=250.44, dep=310.44, wait=119.44
- **UAV-0**: D -> 11 -> D
  - N11: arr=771.70, dep=790.90, wait=27.70
- **UAV-1**: D -> 9 -> D
  - N9: arr=1105.07, dep=1126.67, wait=32.07

---
## Step 4: 插入 Node 10

- **Type**: normal  &emsp; **Coord**: (1.0, 87.0)  &emsp; **Demand**: 43.0 kWh
- **Ready**: 202.0  &emsp; **Due**: 687.0  &emsp; **Service**: 252.0 min

### 當前路徑狀態

- **MCS-0**: D -> 7 -> D &emsp; demand=11, dist=172.00, time=429.89
- **UAV-0**: D -> 11 -> D &emsp; demand=16, dist=49.40, time=818.60
- **UAV-1**: D -> 9 -> D &emsp; demand=18, dist=58.14, time=1158.74

### 候選位置評估

| Route | Pos | Feasible | delta_cost | time_shift | Constraint / Reason |
|:---:|:---:|:---:|---:|---:|:---|
| MCS-0 | 0 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=7, new_arrival=692.10, due_date=618.00, time_shift=441.66) |
| MCS-0 | 1 | OK | 244.56 | 0.00 | FEASIBLE |

### 決策

**插入 MCS-0 位置 1** &emsp; delta_cost = 244.56

### 插入後路徑快照

- **MCS-0**: D -> 7 -> 10 -> D
  - N7: arr=250.44, dep=310.44, wait=119.44
  - N10: arr=446.56, dep=681.10, wait=244.56
- **UAV-0**: D -> 11 -> D
  - N11: arr=771.70, dep=790.90, wait=27.70
- **UAV-1**: D -> 9 -> D
  - N9: arr=1105.07, dep=1126.67, wait=32.07

---
## Step 5: 插入 Node 8

- **Type**: normal  &emsp; **Coord**: (23.0, 2.0)  &emsp; **Demand**: 16.0 kWh
- **Ready**: 167.0  &emsp; **Due**: 724.0  &emsp; **Service**: 97.0 min

### 當前路徑狀態

- **MCS-0**: D -> 7 -> 10 -> D &emsp; demand=54, dist=270.00, time=800.55
- **UAV-0**: D -> 11 -> D &emsp; demand=16, dist=49.40, time=818.60
- **UAV-1**: D -> 9 -> D &emsp; demand=18, dist=58.14, time=1158.74

### 候選位置評估

| Route | Pos | Feasible | delta_cost | time_shift | Constraint / Reason |
|:---:|:---:|:---:|---:|---:|:---|
| MCS-0 | 0 | FAIL | inf | --- | PROPAGATION: service > due (blocked_node=7, new_dep=642.05, due_date=618.00, time_shift=331.61) |
| MCS-0 | 1 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=10, new_arrival=769.94, due_date=687.00, time_shift=323.38) |
| MCS-0 | 2 | FAIL | inf | --- | ARRIVAL > DUE_DATE (actual_depart=681.10, travel=148.61, arrival=829.71, due_date=724.00) |

### 新開路徑比較

| Vehicle | Feasible | Wait (min) | Depart | Travel | Arrival | Mode | Charge Time |
|:---:|:---:|---:|---:|---:|---:|:---:|---:|
| New MCS | OK | 104.17 | 167 | 104.17 | 271.17 | SLOW | 87.27 |

### 決策

**新開 MCS-1**

### 插入後路徑快照

- **MCS-0**: D -> 7 -> 10 -> D
  - N7: arr=250.44, dep=310.44, wait=119.44
  - N10: arr=446.56, dep=681.10, wait=244.56
- **MCS-1**: D -> 8 -> D
  - N8: arr=271.17, dep=358.44, wait=104.17
- **UAV-0**: D -> 11 -> D
  - N11: arr=771.70, dep=790.90, wait=27.70
- **UAV-1**: D -> 9 -> D
  - N9: arr=1105.07, dep=1126.67, wait=32.07

---
## Step 6: 插入 Node 2

- **Type**: normal  &emsp; **Coord**: (51.0, 92.0)  &emsp; **Demand**: 19.0 kWh
- **Ready**: 413.0  &emsp; **Due**: 901.0  &emsp; **Service**: 114.0 min

### 當前路徑狀態

- **MCS-0**: D -> 7 -> 10 -> D &emsp; demand=54, dist=270.00, time=800.55
- **MCS-1**: D -> 8 -> D &emsp; demand=16, dist=150.00, time=462.61
- **UAV-0**: D -> 11 -> D &emsp; demand=16, dist=49.40, time=818.60
- **UAV-1**: D -> 9 -> D &emsp; demand=18, dist=58.14, time=1158.74

### 候選位置評估

| Route | Pos | Feasible | delta_cost | time_shift | Constraint / Reason |
|:---:|:---:|:---:|---:|---:|:---|
| MCS-0 | 0 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=7, new_arrival=636.08, due_date=618.00, time_shift=385.64) |
| MCS-0 | 1 | FAIL | inf | --- | PROPAGATION: service > due (blocked_node=10, new_dep=887.29, due_date=687.00, time_shift=206.19) |
| MCS-0 | 2 | OK | 344.49 | 0.00 | FEASIBLE |
| MCS-1 | 0 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=8, new_arrival=740.25, due_date=724.00, time_shift=469.08) |
| MCS-1 | 1 | OK | 163.89 | 0.00 | FEASIBLE |

### 決策

**插入 MCS-1 位置 1** &emsp; delta_cost = 163.89

### 插入後路徑快照

- **MCS-0**: D -> 7 -> 10 -> D
  - N7: arr=250.44, dep=310.44, wait=119.44
  - N10: arr=446.56, dep=681.10, wait=244.56
- **MCS-1**: D -> 8 -> 2 -> D
  - N8: arr=271.17, dep=358.44, wait=104.17
  - N2: arr=576.89, dep=680.53, wait=163.89
- **UAV-0**: D -> 11 -> D
  - N11: arr=771.70, dep=790.90, wait=27.70
- **UAV-1**: D -> 9 -> D
  - N9: arr=1105.07, dep=1126.67, wait=32.07

---
## Step 7: 插入 Node 3

- **Type**: normal  &emsp; **Coord**: (14.0, 71.0)  &emsp; **Demand**: 17.0 kWh
- **Ready**: 406.0  &emsp; **Due**: 938.0  &emsp; **Service**: 103.0 min

### 當前路徑狀態

- **MCS-0**: D -> 7 -> 10 -> D &emsp; demand=54, dist=270.00, time=800.55
- **MCS-1**: D -> 8 -> 2 -> D &emsp; demand=35, dist=236.00, time=740.25
- **UAV-0**: D -> 11 -> D &emsp; demand=16, dist=49.40, time=818.60
- **UAV-1**: D -> 9 -> D &emsp; demand=18, dist=58.14, time=1158.74

### 候選位置評估

| Route | Pos | Feasible | delta_cost | time_shift | Constraint / Reason |
|:---:|:---:|:---:|---:|---:|:---|
| MCS-0 | 0 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=7, new_arrival=718.17, due_date=618.00, time_shift=467.73) |
| MCS-0 | 1 | FAIL | inf | --- | PROPAGATION: service > due (blocked_node=10, new_dep=913.83, due_date=687.00, time_shift=232.73) |
| MCS-0 | 2 | OK | 315.38 | 0.00 | FEASIBLE |
| MCS-1 | 0 | FAIL | inf | --- | PROPAGATION: service > due (blocked_node=8, new_dep=773.50, due_date=724.00, time_shift=415.06) |
| MCS-1 | 1 | OK | 219.06 | 110.73 | FEASIBLE |
| MCS-1 | 2 | OK | 355.08 | 0.00 | FEASIBLE |

### 決策

**插入 MCS-1 位置 1** &emsp; delta_cost = 219.06

### 插入後路徑快照

- **MCS-0**: D -> 7 -> 10 -> D
  - N7: arr=250.44, dep=310.44, wait=119.44
  - N10: arr=446.56, dep=681.10, wait=244.56
- **MCS-1**: D -> 8 -> 3 -> 2 -> D
  - N8: arr=271.17, dep=358.44, wait=104.17
  - N3: arr=514.33, dep=607.06, wait=108.33
  - N2: arr=687.62, dep=791.25, wait=274.62
- **UAV-0**: D -> 11 -> D
  - N11: arr=771.70, dep=790.90, wait=27.70
- **UAV-1**: D -> 9 -> D
  - N9: arr=1105.07, dep=1126.67, wait=32.07

---
## Step 8: 插入 Node 4

- **Type**: normal  &emsp; **Coord**: (60.0, 20.0)  &emsp; **Demand**: 11.0 kWh
- **Ready**: 418.0  &emsp; **Due**: 989.0  &emsp; **Service**: 68.0 min

### 當前路徑狀態

- **MCS-0**: D -> 7 -> 10 -> D &emsp; demand=54, dist=270.00, time=800.55
- **MCS-1**: D -> 8 -> 3 -> 2 -> D &emsp; demand=52, dist=254.00, time=850.97
- **UAV-0**: D -> 11 -> D &emsp; demand=16, dist=49.40, time=818.60
- **UAV-1**: D -> 9 -> D &emsp; demand=18, dist=58.14, time=1158.74

### 候選位置評估

| Route | Pos | Feasible | delta_cost | time_shift | Constraint / Reason |
|:---:|:---:|:---:|---:|---:|:---|
| MCS-0 | 0 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=7, new_arrival=680.78, due_date=618.00, time_shift=430.33) |
| MCS-0 | 1 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=10, new_arrival=800.22, due_date=687.00, time_shift=353.67) |
| MCS-0 | 2 | OK | 438.10 | 0.00 | FEASIBLE |
| MCS-1 | 0 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=2, new_arrival=978.83, due_date=901.00, time_shift=338.78) |
| MCS-1 | 1 | FAIL | inf | --- | PROPAGATION: service > due (blocked_node=2, new_dep=966.03, due_date=901.00, time_shift=174.78) |
| MCS-1 | 2 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=2, new_arrival=914.28, due_date=901.00, time_shift=226.67) |
| MCS-1 | 3 | OK | 485.75 | 0.00 | FEASIBLE |

### 決策

**插入 MCS-0 位置 2** &emsp; delta_cost = 438.10

### 插入後路徑快照

- **MCS-0**: D -> 7 -> 10 -> 4 -> D
  - N7: arr=250.44, dep=310.44, wait=119.44
  - N10: arr=446.56, dep=681.10, wait=244.56
  - N4: arr=856.10, dep=916.10, wait=438.10
- **MCS-1**: D -> 8 -> 3 -> 2 -> D
  - N8: arr=271.17, dep=358.44, wait=104.17
  - N3: arr=514.33, dep=607.06, wait=108.33
  - N2: arr=687.62, dep=791.25, wait=274.62
- **UAV-0**: D -> 11 -> D
  - N11: arr=771.70, dep=790.90, wait=27.70
- **UAV-1**: D -> 9 -> D
  - N9: arr=1105.07, dep=1126.67, wait=32.07

---
## Step 9: 插入 Node 5

- **Type**: normal  &emsp; **Coord**: (82.0, 86.0)  &emsp; **Demand**: 50.0 kWh
- **Ready**: 531.0  &emsp; **Due**: 1070.0  &emsp; **Service**: 292.0 min

### 當前路徑狀態

- **MCS-0**: D -> 7 -> 10 -> 4 -> D &emsp; demand=65, dist=350.00, time=971.66
- **MCS-1**: D -> 8 -> 3 -> 2 -> D &emsp; demand=52, dist=254.00, time=850.97
- **UAV-0**: D -> 11 -> D &emsp; demand=16, dist=49.40, time=818.60
- **UAV-1**: D -> 9 -> D &emsp; demand=18, dist=58.14, time=1158.74

### 候選位置評估

| Route | Pos | Feasible | delta_cost | time_shift | Constraint / Reason |
|:---:|:---:|:---:|---:|---:|:---|
| MCS-0 | 0 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=7, new_arrival=923.17, due_date=618.00, time_shift=672.73) |
| MCS-0 | 1 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=10, new_arrival=942.62, due_date=687.00, time_shift=496.06) |
| MCS-0 | 2 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=4, new_arrival=1189.94, due_date=989.00, time_shift=333.84) |
| MCS-0 | 3 | FAIL | inf | --- | SERVICE_END > DUE_DATE (arrival=1038.32, charge_time=272.73, departure=1311.05, due_date=1070.00, mode=SLOW) |
| MCS-1 | 0 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=8, new_arrival=1096.78, due_date=724.00, time_shift=825.62) |
| MCS-1 | 1 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=3, new_arrival=1117.62, due_date=938.00, time_shift=603.28) |
| MCS-1 | 2 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=2, new_arrival=1046.45, due_date=901.00, time_shift=358.84) |
| MCS-1 | 3 | FAIL | inf | --- | SERVICE_END > DUE_DATE (arrival=842.64, charge_time=272.73, departure=1115.37, due_date=1070.00, mode=SLOW) |

### 新開路徑比較

| Vehicle | Feasible | Wait (min) | Depart | Travel | Arrival | Mode | Charge Time |
|:---:|:---:|---:|---:|---:|---:|:---:|---:|
| New MCS | OK | 94.44 | 531 | 94.44 | 625.44 | SLOW | 272.73 |

### 決策

**新開 MCS-2**

### 插入後路徑快照

- **MCS-0**: D -> 7 -> 10 -> 4 -> D
  - N7: arr=250.44, dep=310.44, wait=119.44
  - N10: arr=446.56, dep=681.10, wait=244.56
  - N4: arr=856.10, dep=916.10, wait=438.10
- **MCS-1**: D -> 8 -> 3 -> 2 -> D
  - N8: arr=271.17, dep=358.44, wait=104.17
  - N3: arr=514.33, dep=607.06, wait=108.33
  - N2: arr=687.62, dep=791.25, wait=274.62
- **MCS-2**: D -> 5 -> D
  - N5: arr=625.44, dep=898.17, wait=94.44
- **UAV-0**: D -> 11 -> D
  - N11: arr=771.70, dep=790.90, wait=27.70
- **UAV-1**: D -> 9 -> D
  - N9: arr=1105.07, dep=1126.67, wait=32.07

---
## Step 10: 插入 Node 6

- **Type**: normal  &emsp; **Coord**: (74.0, 74.0)  &emsp; **Demand**: 13.0 kWh
- **Ready**: 563.0  &emsp; **Due**: 1086.0  &emsp; **Service**: 80.0 min

### 當前路徑狀態

- **MCS-0**: D -> 7 -> 10 -> 4 -> D &emsp; demand=65, dist=350.00, time=971.66
- **MCS-1**: D -> 8 -> 3 -> 2 -> D &emsp; demand=52, dist=254.00, time=850.97
- **MCS-2**: D -> 5 -> D &emsp; demand=50, dist=136.00, time=992.62
- **UAV-0**: D -> 11 -> D &emsp; demand=16, dist=49.40, time=818.60
- **UAV-1**: D -> 9 -> D &emsp; demand=18, dist=58.14, time=1158.74

### 候選位置評估

| Route | Pos | Feasible | delta_cost | time_shift | Constraint / Reason |
|:---:|:---:|:---:|---:|---:|:---|
| MCS-0 | 0 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=7, new_arrival=753.35, due_date=618.00, time_shift=502.91) |
| MCS-0 | 1 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=10, new_arrival=806.13, due_date=687.00, time_shift=359.58) |
| MCS-0 | 2 | FAIL | inf | --- | PROPAGATION: service > due (blocked_node=4, new_dep=1025.90, due_date=989.00, time_shift=109.80) |
| MCS-0 | 3 | OK | 447.55 | 0.00 | FEASIBLE |
| MCS-1 | 0 | FAIL | inf | --- | PROPAGATION: arrival > due (blocked_node=8, new_arrival=871.41, due_date=724.00, time_shift=600.24) |
| MCS-1 | 1 | FAIL | inf | --- | PROPAGATION: service > due (blocked_node=3, new_dep=984.97, due_date=938.00, time_shift=377.91) |
| MCS-1 | 2 | FAIL | inf | --- | PROPAGATION: service > due (blocked_node=2, new_dep=926.05, due_date=901.00, time_shift=134.80) |
| MCS-1 | 3 | OK | 285.20 | 0.00 | FEASIBLE |
| MCS-2 | 0 | OK | 169.58 | 102.91 | FEASIBLE |
| MCS-2 | 1 | OK | 362.95 | 0.00 | FEASIBLE |

### 決策

**插入 MCS-2 位置 0** &emsp; delta_cost = 169.58

### 插入後路徑快照

- **MCS-0**: D -> 7 -> 10 -> 4 -> D
  - N7: arr=250.44, dep=310.44, wait=119.44
  - N10: arr=446.56, dep=681.10, wait=244.56
  - N4: arr=856.10, dep=916.10, wait=438.10
- **MCS-1**: D -> 8 -> 3 -> 2 -> D
  - N8: arr=271.17, dep=358.44, wait=104.17
  - N3: arr=514.33, dep=607.06, wait=108.33
  - N2: arr=687.62, dep=791.25, wait=274.62
- **MCS-2**: D -> 6 -> 5 -> D
  - N6: arr=629.67, dep=700.58, wait=66.67
  - N5: arr=728.35, dep=1001.08, wait=197.35
- **UAV-0**: D -> 11 -> D
  - N11: arr=771.70, dep=790.90, wait=27.70
- **UAV-1**: D -> 9 -> D
  - N9: arr=1105.07, dep=1126.67, wait=32.07

---
## Final Summary

| Metric | Value |
|---|---|
| MCS Routes | 3 |
| UAV Routes | 2 |
| Total Cost | 169.78 |
| Avg Wait (all) | 161.30 min |
| Total Wait | 1613.00 min |
| Total Distance | 847.53 km |
| Coverage | 100.0% (10/10) |
| Feasible | Yes |

| MCS Fast Charge | 0 |
| MCS Slow Charge | 8 |

### 路徑明細

**MCS-0**: Depot -> N7 -> N10 -> N4 -> Depot  (demand=65, dist=350.00, time=971.66)

| Node | Arrival | Departure | Mode | User Wait |
|:---:|---:|---:|:---:|---:|
| 7 | 250.44 | 310.44 | SLOW | 119.44 |
| 10 | 446.56 | 681.10 | SLOW | 244.56 |
| 4 | 856.10 | 916.10 | SLOW | 438.10 |

**MCS-1**: Depot -> N8 -> N3 -> N2 -> Depot  (demand=52, dist=254.00, time=850.97)

| Node | Arrival | Departure | Mode | User Wait |
|:---:|---:|---:|:---:|---:|
| 8 | 271.17 | 358.44 | SLOW | 104.17 |
| 3 | 514.33 | 607.06 | SLOW | 108.33 |
| 2 | 687.62 | 791.25 | SLOW | 274.62 |

**MCS-2**: Depot -> N6 -> N5 -> Depot  (demand=63, dist=136.00, time=1095.53)

| Node | Arrival | Departure | Mode | User Wait |
|:---:|---:|---:|:---:|---:|
| 6 | 629.67 | 700.58 | SLOW | 66.67 |
| 5 | 728.35 | 1001.08 | SLOW | 197.35 |

**UAV-0**: Depot -> N11 -> Depot  (demand=16, dist=49.40, time=818.60)

| Node | Arrival | Departure | Mode | User Wait |
|:---:|---:|---:|:---:|---:|
| 11 | 771.70 | 790.90 | FAST | 27.70 |

**UAV-1**: Depot -> N9 -> Depot  (demand=18, dist=58.14, time=1158.74)

| Node | Arrival | Departure | Mode | User Wait |
|:---:|---:|---:|:---:|---:|
| 9 | 1105.07 | 1126.67 | FAST | 32.07 |
