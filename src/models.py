import math
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

# --- 1. 基本資料結構 ---

@dataclass
class EVRequest:
    id: int
    x: float                # 將 location 拆開，方便畫圖與存檔
    y: float
    demand: float           # 需電量 (kWh)
    ready_time: float       # 時間窗開始 (Time Window Start)
    due_time: float         # 時間窗結束 (Time Window End)
    service_duration: float # 預估充電作業時間 (min)
    urgency: int            # 優先級 (1-3)
    zone_type: str = "Urban" # 預設為市區，新增此欄位以支援異質場景分析

    # 新增此方法：將物件轉換為字典，以便存入 JSON
    def to_dict(self):
        return asdict(self)

    # 方便取得座標 Tuple 的輔助屬性
    @property
    def location(self):
        return (self.x, self.y)

class Vehicle:
    def __init__(self, id, v_type, capacity, speed, charge_rate, start_loc):
        self.id = id
        self.type = v_type  # 'UAV', 'FastMCS', 'SlowMCS'
        self.capacity = capacity
        self.speed = speed
        self.charge_rate = charge_rate
        self.route = [start_loc] # 紀錄路徑節點 ID 或座標
        self.current_load = capacity
        self.current_time = 0.0

    def calculate_travel_time(self, from_loc: Tuple[float, float], to_loc: Tuple[float, float]) -> float:
        # 子類別需覆寫此方法
        raise NotImplementedError("Subclasses must implement this method")

class UAV(Vehicle):
    def __init__(self, id, start_loc):
        # 假設 UAV 載重限制小 (e.g. 20kWh)，速度快 (60km/h)，充電功率高
        super().__init__(id, 'UAV', capacity=20, speed=60, charge_rate=50, start_loc=start_loc)
    
    def calculate_travel_time(self, from_loc, to_loc):
        # UAV 使用歐幾里得距離 (直線飛行)
        dist = math.sqrt((from_loc[0]-to_loc[0])**2 + (from_loc[1]-to_loc[1])**2)
        # 時間 = 距離 / 速度 * 60 (換算成分鐘，若你的時間單位是分鐘)
        return (dist / self.speed) * 60

class MCS(Vehicle):
    def __init__(self, id, mcs_type, start_loc):
        if mcs_type == 'Fast':
            # 快充車：容量大，充得快
            super().__init__(id, 'FastMCS', capacity=200, speed=40, charge_rate=50, start_loc=start_loc)
        else: 
            # 慢充車：容量大，充得慢
            super().__init__(id, 'SlowMCS', capacity=200, speed=40, charge_rate=10, start_loc=start_loc)

    def calculate_travel_time(self, from_loc, to_loc):
        # 地面車輛使用曼哈頓距離 (模擬街道) * 交通係數
        dist = abs(from_loc[0]-to_loc[0]) + abs(from_loc[1]-to_loc[1])
        # 假設路況係數 1.2 (非直線)
        # 時間 = (距離 * 1.2) / 速度 * 60 (換算成分鐘)
        return (dist * 1.2 / self.speed) * 60