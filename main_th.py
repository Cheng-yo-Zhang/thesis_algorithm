import math
import random
import copy
import numpy as np
import pandas as pd
import time

# 假設你已經將前面寫的類別與函數分別存入對應的檔案：
# 從 alns_core.py 引入 OR 物理環境與算子
from alns_core import Task, MCS, ALNS_State, operator_worst_removal, operator_regret_insertion, operator_random_removal

# =====================================================================
# 1. 任務格式轉換 (還原真實 EV 觸發比例)
# =====================================================================
def prepare_tasks(df_tasks: pd.DataFrame, total_ev_population: int = 500, request_rate: float = 0.25) -> list[Task]:
    actual_requests = int(total_ev_population * request_rate)
    print(f"🔧 1. 系統總共追蹤 {total_ev_population} 台 EV 的軌跡。")
    print(f"   => 實際產生 {actual_requests} 個充電請求，準備分派給 30 台 MCS！")
    
    df_sample = df_tasks.sample(n=actual_requests, random_state=42).sort_values('request_time').reset_index(drop=True)
    base_time = df_sample['request_time'].iloc[0]
    df_sample['relative_sec'] = (df_sample['request_time'] - base_time).dt.total_seconds()
    
    # 恢復快慢充的比例 (例如 70% 緊急救援，30% 日常預約)
    np.random.seed(42)
    request_types = np.random.choice(['Fast', 'Slow'], size=actual_requests, p=[0.7, 0.3])
    
    tasks = []
    for idx, row in df_sample.iterrows():
        req_type = request_types[idx]
        
        # 【變體 B 核心邏輯設定】：
        if req_type == 'Fast':
            tolerable_wait = 600.0  # 緊急拋錨：最多只能等 10 分鐘
            service_time = 900.0    # 統一採用 15 分鐘的超級快充硬體
        else:
            tolerable_wait = 14400.0 # 日常預約：停在公司/家裡，4 小時內來充都可以
            service_time = 900.0     # 同樣是 15 分鐘快充硬體
            
        task = Task(
            task_id=idx + 1,
            req_i=int(row['req_i']),
            req_j=int(row['req_j']),
            request_type=req_type,
            service_time_sec=service_time,
            request_time_sec=row['relative_sec'],
            max_tolerable_wait_sec=tolerable_wait
        )
        tasks.append(task)
        
    print(f"   => 任務物件建立完成！包含 {sum(1 for t in tasks if t.request_type == 'Fast')} 個緊急快充 (10分內) 與 {sum(1 for t in tasks if t.request_type == 'Slow')} 個預約慢充 (4小時內)。")
    return tasks

# =====================================================================
# 2. ALNS 模擬退火主大腦 (Simulated Annealing Meta-heuristic)
# =====================================================================
def run_alns_optimization(initial_state: ALNS_State, iterations: int = 1000) -> ALNS_State:
    print("\n🧠 3. ALNS 最佳化排程大腦啟動 (Simulated Annealing)...")
    
    current_state = copy.deepcopy(initial_state)
    best_state = copy.deepcopy(initial_state)
    
    # 稍微調高初始溫度，讓演算法初期更勇於接受稍微差的解，以利跳出區域最佳
    temperature = 10000.0      
    cooling_rate = 0.99       
    min_temperature = 0.1     
    
    start_time = time.time()
    
    for i in range(iterations):
        # 【修改 1：加大破壞力度 (Large Neighborhood)】
        # 為了能把 100 多台多餘的車裁撤掉，我們必須一次拔除 30% ~ 50% 的任務
        num_to_remove = random.randint(int(500 * 0.3), int(500 * 0.5))
        
        if random.random() < 0.5:
            temp_state = operator_worst_removal(current_state, num_to_remove=num_to_remove)
        else:
            temp_state = operator_random_removal(current_state, num_to_remove=num_to_remove)
        
        # 【修改 2：確保拔除後的任務依照「發生時間」排序，再進行修復】
        # 這能避免未來的任務卡死現在的車輛
        temp_state.unassigned.sort(key=lambda t: t.request_time_sec)
        
        new_state = operator_regret_insertion(temp_state)
        
        current_cost = current_state.get_objective_value()
        new_cost = new_state.get_objective_value()
        
        if new_cost < current_cost:
            current_state = new_state
            if new_cost < best_state.get_objective_value():
                best_state = copy.deepcopy(new_state)
        else:
            import math
            # 避免 math.exp 溢位保護
            try:
                prob = math.exp(-(new_cost - current_cost) / temperature)
            except OverflowError:
                prob = 0.0
            if random.random() < prob:
                current_state = new_state
                
        temperature = max(min_temperature, temperature * cooling_rate)
        
        # 【修改 3：讓報表印出「目前的車隊總數」以利觀察收斂】
        if (i + 1) % 100 == 0:
            obj_val = best_state.get_objective_value()
            fleet_size = len(best_state.fleet)
            extra_cars = max(0, fleet_size - 30)
            print(f"   [Iteration {i+1:04d}] Best Obj: {obj_val:.2f} | 溫度: {temperature:.1f} | 使用車輛數: {fleet_size} (加開 {extra_cars} 台)")
            
    elapsed_time = time.time() - start_time
    print(f"\n✅ ALNS 演算完成！總耗時: {elapsed_time:.2f} 秒")
    return best_state

# =====================================================================
# 3. 主程式執行區
# =====================================================================
if __name__ == "__main__":
    # 使用假資料模擬 24 小時內的請求分佈
    df_dummy = pd.DataFrame({
        'request_time': pd.date_range('2025-01-29 17:00:00', periods=500, freq='86s'),
        'req_i': np.random.randint(20, 80, 500),
        'req_j': np.random.randint(20, 80, 500)
    })
    
    # 【關鍵修改】：傳入 500 台總車輛，並設定 25% 觸發率 (125個任務)
    tasks = prepare_tasks(df_dummy, total_ev_population=500, request_rate=0.25)

    print("\n📡 2. 接收 ConvLSTM 預測熱點，部署 30 台 MCS...")
    mcs_fleet = []
    for idx in range(30):
        # 假設部署在熱區
        mcs_fleet.append(MCS(mcs_id=idx+1, current_i=random.randint(40, 60), current_j=random.randint(40, 60)))

    initial_state = ALNS_State(fleet=mcs_fleet, unassigned=tasks)
    print("   -> 正在產生初始貪婪解 (Initial Greedy Solution)...")
    initial_state = operator_regret_insertion(initial_state)

    # 啟動 ALNS 最佳化
    final_best_state = run_alns_optimization(initial_state, iterations=1000)

    # 報表輸出
    print("\n📊 === [最終論文數據報表] ===")
    served_count = len(tasks) - len(final_best_state.unassigned)
    service_rate = (served_count / len(tasks)) * 100
    
    total_wait_sec = sum(t.actual_wait_time_sec for mcs in final_best_state.fleet for t in mcs.assigned_tasks)
    avg_wait_min = (total_wait_sec / served_count) / 60.0 if served_count > 0 else 0.0
    
    # 計算額外延遲 (包含開車過去會合的時間)
    total_delay_sec = sum(t.actual_extra_delay_sec for mcs in final_best_state.fleet for t in mcs.assigned_tasks)
    avg_delay_min = (total_delay_sec / served_count) / 60.0 if served_count > 0 else 0.0
    
    # 統計 Fast / Slow 任務的表現
    fast_tasks = [t for mcs in final_best_state.fleet for t in mcs.assigned_tasks if t.request_type == 'Fast']
    slow_tasks = [t for mcs in final_best_state.fleet for t in mcs.assigned_tasks if t.request_type == 'Slow']
    
    print(f"模擬系統規模 : 500 台 EV (尖峰時刻測試)")
    print(f"觸發請求數量 : {len(tasks)} 台 IEV")
    print(f"系統服務率   : {service_rate:.1f} %")
    print(f"平均原地等待 : {avg_wait_min:.2f} 分鐘  <-- (EV 在會合點枯等的時間)")
    print(f"平均總體延遲 : {avg_delay_min:.2f} 分鐘  <-- (含 EV 偏離原路線開車會合的時間)")
    print(f"  └ 急件 (Fast) 平均延遲: {(sum(t.actual_extra_delay_sec for t in fast_tasks)/len(fast_tasks)/60) if fast_tasks else 0:.2f} 分鐘")
    print(f"  └ 預約 (Slow) 平均延遲: {(sum(t.actual_extra_delay_sec for t in slow_tasks)/len(slow_tasks)/60) if slow_tasks else 0:.2f} 分鐘")