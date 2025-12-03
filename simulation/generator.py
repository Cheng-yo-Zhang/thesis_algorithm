# simulation/generator.py
import numpy as np
from config.settings import *
from core.models import Request # <--- 修正引用

class RequestGenerator:
    def __init__(self):
        pass

    def _calculate_dynamic_remoteness(self, req_x, req_y, chargers):
        """計算到最近地面車隊 (Fast/Slow MCS) 的曼哈頓距離"""
        min_dist = float('inf')
        ground_chargers = [c for c in chargers if 'MCS' in c.type]
        
        if not ground_chargers: return float('inf')

        for mcs in ground_chargers:
            dist = abs(req_x - mcs.x) + abs(req_y - mcs.y)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def generate_requests(self, time_slot, chargers, poisson_lambda=25):
        # 泊松分佈決定數量
        num_requests = max(5, min(50, np.random.poisson(poisson_lambda)))
        
        # 隨機生成全域路況 (30% 機率塞車)
        current_traffic = np.random.uniform(0.5, 1.5) if np.random.random() < 0.3 else 0.0
        
        requests = []
        for i in range(num_requests):
            req = self._create_single_request(time_slot, i, current_traffic, chargers)
            requests.append(req)
        return requests, current_traffic

    def _create_single_request(self, time_slot, idx, traffic_val, chargers):
        # 1. 位置生成
        if np.random.random() < PROB_GENERATION_IN_HOT:
            z = GENERATION_HOT_ZONES[np.random.randint(0, len(GENERATION_HOT_ZONES))]
            px = np.clip(np.random.normal(z['center'][0], z['sigma']), 0, GRID_SIZE)
            py = np.clip(np.random.normal(z['center'][1], z['sigma']), 0, GRID_SIZE)
        else:
            px = np.random.uniform(0, GRID_SIZE)
            py = np.random.uniform(0, GRID_SIZE)

        # 2. 狀態生成
        soc = np.random.uniform(0.01, REQUEST_THRESHOLD_SOC)
        if np.random.random() < PROB_SHORT_STAY:
            stay = np.random.randint(*TIME_WINDOW_FAST)
        else:
            stay = np.random.randint(*TIME_WINDOW_SLOW)

        req = Request(f"{time_slot}_{idx}", px, py, soc, stay, traffic_val)
        
        # 3. 分類 (傳入 chargers 進行動態偏遠判定)
        self._classify_request_type(req, chargers)
        return req

    def _classify_request_type(self, req, chargers):
        # 判定 1: 動態偏遠 (相對於 MCS 車隊位置)
        dist_to_mcs = self._calculate_dynamic_remoteness(req.x, req.y, chargers)
        
        if dist_to_mcs > REMOTE_DISTANCE_THRESHOLD:
            req.req_type = 'URGENT_UAV'
            req.max_tolerance = 300
            req.note = f"Remote (Dist:{dist_to_mcs:.0f})"
            return

        # 判定 2 & 3: 時間窗 -> 快/慢充
        if req.stay_duration < 60:
            req.req_type = 'FAST_MCS'
            req.max_tolerance = 600
        else:
            req.req_type = 'SLOW_MCS'
            req.max_tolerance = 3600