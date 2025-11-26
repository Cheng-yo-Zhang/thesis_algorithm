import json
import os
import random

# [修正] 強制使用 Agg 後端，解決 Windows Tcl/Tk 錯誤
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from src.models import EVRequest

class DatasetGenerator:
    def __init__(self, num_requests=50, area_size=100):
        self.num_requests = num_requests
        self.area_size = area_size
        self.depot_loc = (50, 50)
        self.requests = []

    def generate_scenario(self):
        requests = []
        req_id = 1
        
        # 1. 偏遠景區 (Remote)
        num_remote = int(self.num_requests * 0.2)
        for _ in range(num_remote):
            if random.random() > 0.5:
                cx, cy = random.uniform(80, 100), random.uniform(80, 100)
            else:
                cx, cy = random.uniform(0, 20), random.uniform(0, 20)
                
            req = EVRequest(
                id=req_id,
                x=cx,
                y=cy,
                demand=random.uniform(5, 15), 
                ready_time=random.randint(0, 60),
                due_time=random.randint(90, 180), 
                service_duration=15,
                zone_type="Remote",
                urgency=3
            )
            requests.append(req)
            req_id += 1

        # 2. 市中心 (Urban)
        num_urban = int(self.num_requests * 0.5)
        for _ in range(num_urban):
            cx = random.gauss(50, 15)
            cy = random.gauss(50, 15)
            cx = max(0, min(self.area_size, cx))
            cy = max(0, min(self.area_size, cy))
            
            req = EVRequest(
                id=req_id,
                x=cx,
                y=cy,
                demand=random.uniform(30, 60), 
                ready_time=random.randint(0, 120),
                due_time=random.randint(300, 480), 
                service_duration=random.randint(30, 60),
                zone_type="Urban",
                urgency=1
            )
            requests.append(req)
            req_id += 1

        # 3. 物流園區 (Logistics)
        num_logistic = self.num_requests - num_remote - num_urban
        park_x, park_y = 20, 80 
        for _ in range(num_logistic):
            cx = random.gauss(park_x, 5)
            cy = random.gauss(park_y, 5)
            
            req = EVRequest(
                id=req_id,
                x=cx,
                y=cy,
                demand=random.uniform(20, 40),
                ready_time=240, 
                due_time=480,
                service_duration=30,
                zone_type="Logistics",
                urgency=2
            )
            requests.append(req)
            req_id += 1

        self.requests = requests
        return requests

    def save_to_json(self, filename="data/scenario_01.json"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data_to_save = {
            "metadata": {
                "num_requests": self.num_requests,
                "area_size": self.area_size,
                "depot_loc": self.depot_loc
            },
            "requests": [req.to_dict() for req in self.requests]
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"資料已成功儲存至: {filename}")

    def plot_map(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.depot_loc[0], self.depot_loc[1], c='black', s=200, marker='*', label='Depot')
        
        groups = [("Urban", 'blue', 'o'), ("Remote", 'red', '^'), ("Logistics", 'green', 's')]
        for zone, color, marker in groups:
            x_vals = [r.x for r in self.requests if r.zone_type == zone]
            y_vals = [r.y for r in self.requests if r.zone_type == zone]
            if x_vals:
                plt.scatter(x_vals, y_vals, c=color, marker=marker, label=zone, alpha=0.7)

        plt.title(f"Simulation Scenario: {self.num_requests} Requests")
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        plt.xlim(0, self.area_size)
        plt.ylim(0, self.area_size)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # [修正] 改為存檔
        output_img = "data/scenario_viz.png"
        plt.savefig(output_img)
        print(f"分佈圖已儲存至: {output_img}")
        plt.close()

if __name__ == "__main__":
    generator = DatasetGenerator(num_requests=50)
    data = generator.generate_scenario()
    generator.save_to_json("data/test_scenario_50.json")
    generator.plot_map()