import math
import random
from dataclasses import dataclass
from typing import List, Tuple

# --- 1. 基本資料結構 ---

@dataclass
class EVRequest:
    id: int
    location: Tuple[float, float]  # (x, y)
    demand: float                  # 需電量
    time_window: Tuple[float, float] # (start, end)
    service_duration: float        # 預估充電所需時間 (會隨服務車種變動，暫存基礎值)
    urgency: int                   # 優先級

class Vehicle:
    def __init__(self, id, v_type, capacity, speed, charge_rate, start_loc):
        self.id = id
        self.type = v_type  # 'UAV', 'FastMCS', 'SlowMCS'
        self.capacity = capacity
        self.speed = speed
        self.charge_rate = charge_rate
        self.route = [start_loc] # 紀錄路徑
        self.current_load = capacity
        self.current_time = 0.0

    def calculate_travel_time(self, from_loc, to_loc):
        # 子類別需覆寫此方法 (UAV vs MCS)
        pass

class UAV(Vehicle):
    def __init__(self, id, start_loc):
        # 假設 UAV 載重限制小，速度快，充電功率高
        super().__init__(id, 'UAV', capacity=20, speed=60, charge_rate=50, start_loc=start_loc)
    
    def calculate_travel_time(self, from_loc, to_loc):
        # 歐幾里得距離 / 速度
        dist = math.sqrt((from_loc[0]-to_loc[0])**2 + (from_loc[1]-to_loc[1])**2)
        return dist / self.speed

class MCS(Vehicle):
    def __init__(self, id, mcs_type, start_loc):
        if mcs_type == 'Fast':
            super().__init__(id, 'FastMCS', capacity=200, speed=40, charge_rate=50, start_loc=start_loc)
        else: # Slow
            super().__init__(id, 'SlowMCS', capacity=200, speed=40, charge_rate=10, start_loc=start_loc)

    def calculate_travel_time(self, from_loc, to_loc):
        # 曼哈頓距離 (模擬街道) / 速度 * 交通係數
        dist = abs(from_loc[0]-to_loc[0]) + abs(from_loc[1]-to_loc[1])
        return (dist * 1.2) / self.speed # 假設路況係數 1.2

# --- 2. 初始解生成 (Constructive Heuristic) ---
# 這裡建議先用簡單的 Greedy 策略：誰近誰先服務，滿足時間窗即可

def generate_initial_solution(requests: List[EVRequest], fleet: List[Vehicle]):
    # 簡單範例：這是一個非常粗糙的貪婪法，你需要優化這裡
    unassigned = requests.copy()
    
    for vehicle in fleet:
        if not unassigned: break
        
        # 嘗試分配任務
        # TODO: 這裡需要加入你的 heuristic 邏輯
        # 例如：計算 cost = 距離 + 時間窗懲罰
        # 選擇 cost 最小的 request 加入 vehicle.route
        pass
    
    return fleet

# --- 3. 主要演算法 (Meta-heuristic) ---
# 針對這種複雜問題，建議使用 ALNS (Adaptive Large Neighborhood Search) 或 GA

def run_optimization(requests, fleet):
    current_solution = generate_initial_solution(requests, fleet)
    
    # 迭代優化迴圈
    max_iterations = 1000
    for i in range(max_iterations):
        # 1. Destroy: 隨機移除幾個任務
        # 2. Repair: 用比較好的邏輯重新插入任務 (考慮 UAV 適合遠且急的，MCS 適合近且大的)
        # 3. 評估 Objective Function (總服務時間 + 懲罰項)
        pass
        
    return current_solution