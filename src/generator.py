import json
import os
import random
import math
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from src.models import EVRequest

class DatasetGenerator:
    def __init__(self, num_requests=100, area_size=100):
        self.num_requests = num_requests
        self.area_size = area_size
        self.depot_loc = (0, 0)
        self.requests = []

    def _generate_request_attributes(self, zone_type):
        """
        根據區域類型 (Cold/Hot) 強制決定 Request 的具體屬性
        """
        # [規則設定]
        # 冷門區域 (Cold) -> 強制 Type A (UAV)
        # 熱門區域 (Hot)  -> 強制 Type B (Fast Truck) 或 Type C (Slow Truck)
        
        if zone_type == "Cold":
            # --- 冷門區域：全交給 UAV ---
            req_type = "TypeA"
            demand = random.uniform(5, 12)       # 小電量 (UAV 載重限制)
            service_duration = 15                # 快充
            ready_time = random.randint(0, 60)
            due_time = ready_time + random.randint(30, 60) # 時間窗緊迫
            urgency = 3
            charge_speed = "FAST" # UAV 基本上是快充補給
            
        else: # Hot Zone (熱門區域)
            # --- 熱門區域：地面車隊 (快充車 vs 慢充車) ---
            # 假設 40% 是趕時間的快充需求，60% 是可過夜的慢充需求
            req_type = random.choices(["TypeB", "TypeC"], weights=[0.4, 0.6])[0]
            
            if req_type == "TypeB": # 快充車
                demand = random.uniform(30, 60)      # 大電量
                service_duration = 45                # 服務時間中等
                ready_time = random.randint(0, 120)
                due_time = ready_time + random.randint(60, 90) # 時間窗稍緊
                urgency = 2
                charge_speed = "FAST"
            else: # TypeC: 慢充車
                demand = random.uniform(20, 50)
                service_duration = 120               # 服務時間長
                ready_time = random.randint(0, 240)
                due_time = ready_time + random.randint(300, 480) # 時間窗極寬
                urgency = 1
                charge_speed = "SLOW"

        return demand, service_duration, ready_time, due_time, urgency, charge_speed, req_type

    def generate_scenario(self):
        # 1. 隨機撒點 & DBSCAN 分群 (維持 DBSCAN 參數)
        # eps=8, min_samples=3 是一個很好的現實比例
        coords = np.random.uniform(5, self.area_size - 5, (self.num_requests, 2))
        db = DBSCAN(eps=8, min_samples=3).fit(coords)
        labels = db.labels_ 
        
        requests = []
        
        for i, (x, y) in enumerate(coords):
            label = labels[i]
            req_id = i + 1
            dist_from_depot = math.sqrt((x - self.depot_loc[0])**2 + (y - self.depot_loc[1])**2)
            
            # [區域判定邏輯]
            # Cold (冷門): 既孤立 (Noise) 又遠 (>25km)
            # Hot (熱門): 聚落 (Cluster) 或 離 Depot 近
            if label == -1 and dist_from_depot > 25:
                zone_type = "Cold"
            else:
                zone_type = "Hot"
            
            # [屬性生成]
            demand, duration, r_time, d_time, urgency, speed, r_type = self._generate_request_attributes(zone_type)
            
            req = EVRequest(
                id=req_id,
                x=float(x), y=float(y),
                demand=demand,
                ready_time=r_time,
                due_time=d_time,
                service_duration=duration,
                zone_type=zone_type,  
                urgency=urgency,
                charge_type=speed,
                resource_fit=r_type
            )
            # 記錄額外資訊方便繪圖與驗證
            req.charge_type = speed   
            req.resource_fit = r_type 
            
            requests.append(req)

        self.requests = requests
        
        # 簡單統計
        count_cold = len([r for r in requests if r.zone_type == "Cold"])
        count_hot = len([r for r in requests if r.zone_type == "Hot"])
        print(f"場景生成完畢 (N={self.num_requests})")
        print(f"  - 冷門區域 (Cold/UAV): {count_cold}")
        print(f"  - 熱門區域 (Hot/Truck): {count_hot}")
        
        return requests

    def save_to_json(self, filename="data/scenario_strict.json"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data_to_save = {
            "metadata": {
                "num_requests": self.num_requests,
                "area_size": self.area_size,
                "generation_rule": "Strict Division: Cold->UAV, Hot->Trucks"
            },
            "requests": [req.to_dict() for req in self.requests]
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"資料已儲存至: {filename}")

    def plot_map(self):
        plt.figure(figsize=(10, 10))
        # 畫 Depot
        plt.scatter(self.depot_loc[0], self.depot_loc[1], c='black', s=300, marker='*', label='Depot')
        
        # 畫 25km 保護圈
        circle = plt.Circle(self.depot_loc, 25, color='gray', fill=False, linestyle=':', alpha=0.5)
        plt.gca().add_patch(circle)

        # 繪製各種類型的點
        # 邏輯：
        # Cold Zone -> 一定是 Type A (UAV) -> 紅色三角形
        # Hot Zone -> 可能是 Type B (快充車/藍色圓形) 或 Type C (慢充車/藍色方塊)
        
        for r in self.requests:
            if r.zone_type == 'Cold':
                color = 'red' 
                marker = '^' # 三角形代表 UAV
            else:
                color = 'blue'
                if r.resource_fit == 'TypeB':
                    marker = 'o' # 圓形代表快充車
                else:
                    marker = 's' # 方塊代表慢充車
            
            plt.scatter(r.x, r.y, c=color, marker=marker, alpha=0.7)

        # 自定義圖例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='k', markersize=15, label='Depot'),
            Line2D([0], [0], linestyle=':', color='gray', label='Near Depot (<25km)'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Cold Zone: UAV Only (Type A)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Hot Zone: Fast Truck (Type B)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Hot Zone: Slow Truck (Type C)'),
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.title(f"Scenario: Strict Resource Allocation (Cold=UAV, Hot=Trucks)")
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        plt.xlim(-5, self.area_size + 5)
        plt.ylim(-5, self.area_size + 5)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        output_img = "data/scenario_viz_strict.png"
        plt.savefig(output_img)
        print(f"分佈圖已儲存至: {output_img}")
        plt.close()

if __name__ == "__main__":
    gen = DatasetGenerator(100)
    gen.generate_scenario()
    gen.save_to_json()
    gen.plot_map()