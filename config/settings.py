# config/settings.py
import numpy as np

# ==========================================
# 1. 基礎模擬環境
# ==========================================
GRID_SIZE = 100            # 網格大小 100x100 [cite: 500]
ROAD_LENGTH = 0.5          # 路段長度 0.5 km [cite: 500]
TIME_SLOT_DURATION = 60    # 60 秒 [cite: 480]
TOTAL_TIME_SLOTS = 1440    # 24 小時 [cite: 480]

DEPOT_LOCATION = (0, 0)    # MCS 基地 [cite: 504]
FCS_LOCATIONS = [(20, 20), (80, 20), (50, 80)] # [cite: 503]

# ==========================================
# 2. 電動車 (EV) 參數
# ==========================================
TOTAL_EVS = 500            # [cite: 480]
EV_SPEED = 15.0            # 15 m/s [cite: 480]
EV_BATTERY_CAPACITY = 90.0 # 90 kWh [cite: 480]

TARGET_CHARGING_PERCENT = 0.40 
AMOUNT_NEEDED_PER_CHARGE = EV_BATTERY_CAPACITY * TARGET_CHARGING_PERCENT # 36.0 kWh [cite: 128]

# 時間窗定義 (分鐘)
TIME_WINDOW_FAST = (10, 60)    
TIME_WINDOW_SLOW = (240, 480)  

REQUEST_THRESHOLD_SOC = 0.20  # [cite: 127]

# ==========================================
# 3. 異質充電資源配置
# ==========================================
CHARGER_CONFIG = {
    'UAV': {
        'count': 5,            
        'speed': 25.0,         
        'capacity': 20.0,      
        'power': 120.0,        
        'movement_type': 'euclidean', 
    },
    'FastMCS': {
        'count': 10,
        'speed': 12.0,         # [cite: 480]
        'capacity': 270.0,     # [cite: 480]
        'power': 330.0,        # [cite: 480]
        'movement_type': 'manhattan', 
    },
    'SlowMCS': {
        'count': 15,
        'speed': 10.0,         
        'capacity': 300.0,     
        'power': 60.0,         
        'movement_type': 'manhattan',
    }
}

# ==========================================
# 4. 判定邏輯閾值
# ==========================================
REMOTE_DISTANCE_THRESHOLD = 40  # 曼哈頓距離閾值

# ==========================================
# 5. 生成分佈設定
# ==========================================
GENERATION_HOT_ZONES = [
    {'center': (30, 30), 'sigma': 15}, 
    {'center': (70, 70), 'sigma': 15}
]
PROB_GENERATION_IN_HOT = 0.7 
PROB_SHORT_STAY = 0.6