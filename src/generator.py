import json
import os
import random
import math  # 需要引入 math 計算距離
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

    def generate_scenario(self):
        # 1. 隨機撒點
        coords = np.random.uniform(5, self.area_size - 5, (self.num_requests, 2))
        
        # 2. DBSCAN 分群
        db = DBSCAN(eps=8, min_samples=3).fit(coords)
        labels = db.labels_ 
        
        requests = []
        
        for i, (x, y) in enumerate(coords):
            label = labels[i]
            req_id = i + 1
            
            # 計算該點到 Depot (0,0) 的距離
            dist_from_depot = math.sqrt((x - self.depot_loc[0])**2 + (y - self.depot_loc[1])**2)
            
            # [修正邏輯]
            # 判定為 Remote 的條件：
            # 1. 是 DBSCAN 的噪音點 (label == -1)
            # 2. 且距離 Depot 超過 25 公里 (避免家門口的點被算成偏遠)
            if label == -1 and dist_from_depot > 25:
                # --- 真．偏遠地區 ---
                zone_type = "Remote"
                demand = random.uniform(5, 15)
                ready_time = random.randint(0, 60)
                due_time = random.randint(90, 180)
                service_duration = 15
                urgency = 3
            else:
                # --- 市區 (包含聚落點 以及 離家近的孤單點) ---
                zone_type = "Urban"
                demand = random.uniform(30, 60)
                ready_time = random.randint(0, 120)
                due_time = random.randint(300, 480)
                service_duration = random.randint(30, 60)
                urgency = 1
            
            req = EVRequest(
                id=req_id,
                x=float(x), y=float(y),
                demand=demand,
                ready_time=ready_time,
                due_time=due_time,
                service_duration=service_duration,
                zone_type=zone_type,
                urgency=urgency
            )
            requests.append(req)

        self.requests = requests
        
        # 統計
        num_remote = len([r for r in requests if r.zone_type == "Remote"])
        num_urban = len(requests) - num_remote
        print(f"場景生成完畢 (N={self.num_requests})")
        print(f"  - 市中心 (Urban): {num_urban}")
        print(f"  - 偏遠區 (Remote): {num_remote} (已排除距離 Depot < 25km 的點)")
        
        return requests

    def save_to_json(self, filename="data/scenario_final.json"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data_to_save = {
            "metadata": {
                "num_requests": self.num_requests,
                "area_size": self.area_size,
                "depot_loc": self.depot_loc,
                "generation_method": "DBSCAN + Distance Filter (>25km)"
            },
            "requests": [req.to_dict() for req in self.requests]
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"資料已儲存至: {filename}")

    def plot_map(self):
        plt.figure(figsize=(10, 10))
        
        # 畫 Depot
        plt.scatter(self.depot_loc[0], self.depot_loc[1], c='black', s=250, marker='*', label='Depot (0,0)')
        
        # 畫半徑 25km 的圈圈 (視覺輔助，說明為什麼這裡沒紅點)
        circle = plt.Circle(self.depot_loc, 25, color='gray', fill=False, linestyle=':', alpha=0.5, label='Near Depot (<25km)')
        plt.gca().add_patch(circle)

        groups = [("Urban", 'blue', 'o'), ("Remote", 'red', '^')]
        
        for zone, color, marker in groups:
            x_vals = [r.x for r in self.requests if r.zone_type == zone]
            y_vals = [r.y for r in self.requests if r.zone_type == zone]
            if x_vals:
                plt.scatter(x_vals, y_vals, c=color, marker=marker, label=f"{zone} ({len(x_vals)})", alpha=0.7)

        plt.title(f"Final Scenario: Distance Constrained Remote Areas")
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        plt.xlim(-5, self.area_size + 5)
        plt.ylim(-5, self.area_size + 5)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        output_img = "data/scenario_viz_final.png"
        plt.savefig(output_img)
        print(f"分佈圖已儲存至: {output_img}")
        plt.close()

if __name__ == "__main__":
    generator = DatasetGenerator(num_requests=100)
    generator.generate_scenario()
    generator.save_to_json()
    generator.plot_map()