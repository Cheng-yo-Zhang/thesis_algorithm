# core/models.py
from config import settings

class Request:
    def __init__(self, req_id, x, y, soc, stay_duration, traffic_val=0.0):
        self.id = req_id
        self.x = x
        self.y = y
        self.soc = soc
        self.stay_duration = stay_duration
        self.traffic_val = traffic_val
        
        # 分類結果
        self.req_type = None
        self.max_tolerance = 0 
        self.note = ""

        # [新增] 狀態追蹤
        self.is_served = False 
        # 計算 Urgency 用 (Deadline = Generate Time + Tolerance)
        # 這裡為了簡化，我們先在 Generator 賦予它一個具體的 due_time，
        # 或是之後在 main.py 裡動態計算。目前先用 generator 產生的屬性。
        self.due_time = 0 

    @property
    def energy_demand(self):
        """計算需要充電的能量 (kWh)"""
        # 從當前 SOC 充到目標百分比所需的能量
        target_soc = self.soc + settings.TARGET_CHARGING_PERCENT
        target_soc = min(target_soc, 1.0)  # 不超過 100%
        return (target_soc - self.soc) * settings.EV_BATTERY_CAPACITY

    def __repr__(self):
        status = "Done" if self.is_served else "Pending"
        return f"<Req {self.id} | {self.req_type} | {status}>"

class Charger:
    def __init__(self, charger_id, start_x, start_y, type_name):
        self.id = charger_id
        self.type = type_name
        self.x = start_x
        self.y = start_y
        
        # [修改] 狀態機擴充
        # IDLE: 閒置, MOVING: 移動中, SERVING: 服務中
        self.status = 'IDLE' 
        
        # [新增] 導航目標
        self.target_x = None
        self.target_y = None
        
        cfg = settings.CHARGER_CONFIG[type_name]
        self.speed = cfg['speed']
        self.capacity = cfg['capacity']
        self.power = cfg['power']
        self.current_charging_power = 0.0
        self.move_mode = cfg['movement_type']
        
        self.current_energy = self.capacity

    def set_target(self, x, y):
        """指派移動目標"""
        self.target_x = x
        self.target_y = y
        self.status = 'MOVING'

    def clear_target(self):
        """清除目標 (到達或取消)"""
        self.target_x = None
        self.target_y = None
        self.status = 'IDLE'

    def __repr__(self):
        return f"<{self.type} #{self.id} | {self.status} | SoC:{self.current_energy:.1f}>"