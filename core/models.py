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
        
        # 分類結果 (由 generator 填入)
        self.req_type = None   # 'URGENT_UAV', 'FAST_MCS', 'SLOW_MCS'
        self.max_tolerance = 0 
        self.note = ""         # 備註 (例如: Remote)

    def __repr__(self):
        return f"<Req {self.id} | {self.req_type} | Loc:({self.x:.1f},{self.y:.1f})>"

class Charger:
    def __init__(self, charger_id, start_x, start_y, type_name):
        self.id = charger_id
        self.type = type_name
        self.x = start_x
        self.y = start_y
        self.status = 'IDLE' # IDLE, MOVING, CHARGING
        
        # 從設定檔讀取規格
        cfg = settings.CHARGER_CONFIG[type_name]
        self.speed = cfg['speed']
        self.capacity = cfg['capacity']
        self.power = cfg['power']
        self.move_mode = cfg['movement_type']
        
        self.current_energy = self.capacity # 初始滿電

    def __repr__(self):
        return f"<{self.type} #{self.id} at ({self.x:.1f}, {self.y:.1f})>"