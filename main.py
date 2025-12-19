import random
import os
import json
import math
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import numpy as np


class VRPDataGenerator:
    """
    VRP 資料生成器：生成包含 MCS 和 UAV 混合車隊的車輛路徑問題測試案例。
    
    Attributes:
        map_size: 地圖大小 (km)
        num_requests: 客戶需求數量
    """
    
    # ==================== 常數定義 ====================
    # 時間常數 (分鐘)
    MINUTES_PER_DAY = 1440
    WORK_START_TIME = 480   # 08:00
    WORK_END_TIME = 1080    # 18:00
    
    # MCS 參數
    SPEED_MCS = (12 * 3.6) / 60.0  # km/min
    CAPACITY_MCS = 270             # kWh
    POWER_FAST = 250               # kW (Super Fast Charging)
    POWER_SLOW = 11                 # kW (AC Slow Charging)
    
    # UAV 參數
    SPEED_UAV = 60 / 60.0          # 1 km/min (60 km/h)
    CAPACITY_UAV = 15              # kWh
    POWER_UAV_FAST = 50            # kW
    UAV_MAX_PAYLOAD = 12.0         # kWh，無人機最大載電量
    
    # 機率與地形參數
    PROB_RESTRICTED = 0.1          # 受限區域機率
    PROB_FAST = 0.3                # 快充需求機率
    CLUSTER_RADIUS_RATIO = 0.15    # 受限區域半徑比例
    
    # 服務時間附加 (分鐘)
    SERVICE_OVERHEAD_UAV = 10.0
    SERVICE_OVERHEAD_NORMAL = 5.0
    
    def __init__(self, map_size: int = 100, num_requests: Optional[int] = None, seed: Optional[int] = None):
        """
        初始化 VRP 資料生成器。
        
        Args:
            map_size: 地圖大小 (km)，預設 100
            num_requests: 客戶需求數量，若為 None 則隨機生成 5-50
            seed: 隨機種子，用於結果重現
        """
        if seed is not None:
            random.seed(seed)
        
        self.map_size = map_size
        self.num_requests = num_requests if num_requests else random.randint(5, 50)
        
        if self.num_requests <= 0:
            raise ValueError("num_requests 必須大於 0")

        self.nodes: List[Dict[str, Any]] = []
        self.fleet: Dict[str, Any] = {}
        self.cluster_centers: List[tuple] = []  # 受限區域中心點

    def _generate_cluster_centers(self) -> None:
        """生成受限區域的中心點（在生成需求前調用一次）"""
        num_clusters = random.randint(2, 5)
        self.cluster_centers = []
        for _ in range(num_clusters):
            cx = random.uniform(self.map_size * 0.1, self.map_size * 0.9)
            cy = random.uniform(self.map_size * 0.1, self.map_size * 0.9)
            self.cluster_centers.append((cx, cy))

    def _is_in_restricted_area(self, x: float, y: float) -> bool:
        """
        判斷座標是否在受限區域內。
        
        Args:
            x: X 座標
            y: Y 座標
            
        Returns:
            是否在受限區域內
        """
        for cx, cy in self.cluster_centers:
            distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if distance < self.map_size * self.CLUSTER_RADIUS_RATIO:
                return True
        return False

    def _calculate_service_time(self, demand: float, terrain: int, req_type: str) -> float:
        """
        計算服務時間。
        
        Args:
            demand: 需求量 (kWh)
            terrain: 地形類型 (0: 一般, 1: 受限)
            req_type: 需求類型 ("FAST" / "SLOW")
            
        Returns:
            服務時間 (分鐘)
        """
        if terrain == 1:
            power = self.POWER_UAV_FAST
            overhead = self.SERVICE_OVERHEAD_UAV
        else:
            power = self.POWER_FAST if req_type == "FAST" else self.POWER_SLOW
            overhead = self.SERVICE_OVERHEAD_NORMAL
        
        return round((demand / power) * 60 + overhead, 1)

    def generate_requests(self) -> None:
        """
        生成需求點數據。
        採用兩階段生成法，確保 '受限區域' 算在 '快充' 總名額內。
        """
        # 1. 初始化與生成 Depot
        self._generate_cluster_centers()
        self.nodes = []
        depot = {
            "id": 0, "type": "DEPOT", "x": 0, "y": 0, "demand": 0,
            "r_time": 0, "l_time": self.MINUTES_PER_DAY, "terrain": 0, "service_time": 0
        }
        self.nodes.append(depot)

        # 2. 計算目標數量 (總量控制)
        target_fast_count = int(self.num_requests * self.PROB_FAST) # 例如 20 * 0.3 = 6
        
        # 3. 第一階段：先生成所有座標與地形 (暫存起來，還不決定詳細屬性)
        temp_nodes = []
        for i in range(1, self.num_requests + 1):
            x = random.randint(0, self.map_size)
            y = random.randint(0, self.map_size)
            terrain = 1 if self._is_in_restricted_area(x, y) else 0
            temp_nodes.append({
                "id": i,
                "x": x,
                "y": y,
                "terrain": terrain
            })

        # 4. 第二階段：分配類型 (Type Assignment)
        # 先分組
        restricted_group = [n for n in temp_nodes if n['terrain'] == 1]
        normal_group = [n for n in temp_nodes if n['terrain'] == 0]
        
        # A. 受限區域強制為 FAST (這就是你要的：算在快速充電裡面)
        for node in restricted_group:
            node['type'] = 'FAST'
            
        # B. 計算還剩下多少快充名額
        slots_used = len(restricted_group)
        slots_remaining = target_fast_count - slots_used
        
        # C. 處理一般區域的分配
        if slots_remaining > 0:
            # 如果還有名額，從一般區域隨機挑選幸運兒變成 FAST
            # min 是為了防止剩下的名額比一般節點還多 (雖然機率極低)
            count_to_pick = min(slots_remaining, len(normal_group))
            
            # 隨機抽出要變成快充的 index
            fast_indices = set(random.sample(range(len(normal_group)), count_to_pick))
            
            for idx, node in enumerate(normal_group):
                if idx in fast_indices:
                    node['type'] = 'FAST'
                else:
                    node['type'] = 'SLOW'
        else:
            # 如果受限區域已經太多，佔滿(或超過)了快充名額，剩下的一般區域只能全是 SLOW
            for node in normal_group:
                node['type'] = 'SLOW'

        # 將分組後的節點合併回來 (依照 ID 排序以保持順序)
        processed_nodes = restricted_group + normal_group
        processed_nodes.sort(key=lambda x: x['id'])

        # 5. 第三階段：根據已確定的類型，生成需求量與時間窗
        for node in processed_nodes:
            req_type = node['type']
            terrain = node['terrain']
            
            # 決定需求量
            if terrain == 1:
                demand = round(random.uniform(4.0, self.UAV_MAX_PAYLOAD), 1)
            elif req_type == "FAST":
                demand = round(random.uniform(12, 40), 1)
            else: # SLOW
                demand = round(random.uniform(20, 80), 1)
            
            service_time = self._calculate_service_time(demand, terrain, req_type)
            request_time = random.randint(self.WORK_START_TIME, self.WORK_END_TIME)
            
            if req_type == "FAST":
                tolerance = random.randint(30, 60)
            else:
                tolerance = random.randint(60, 180)
            
            deadline = request_time + tolerance

            # 構建最終節點並加入 self.nodes
            full_node = {
                "id": node['id'],
                "type": req_type,
                "x": node['x'],
                "y": node['y'],
                "demand": demand,
                "r_time": request_time,
                "l_time": deadline,
                "terrain": terrain,
                "service_time": service_time
            }
            self.nodes.append(full_node)

    def generate_fleet(self) -> None:
        """定義車隊參數"""
        self.fleet = {
            "UAV": {
                "speed": self.SPEED_UAV,
                "capacity": self.CAPACITY_UAV,
                "can_serve_restricted": True,
                "compatible_types": ["FAST"],
                "count": 2
            },
            "MCS_FAST": {
                "speed": self.SPEED_MCS,
                "capacity": self.CAPACITY_MCS,
                "can_serve_restricted": False,
                "compatible_types": ["FAST"],
                "count": 3
            },
            "MCS_SLOW": {
                "speed": self.SPEED_MCS,
                "capacity": self.CAPACITY_MCS,
                "can_serve_restricted": False,
                "compatible_types": ["SLOW"],
                "count": 5
            }
        }

    def save_to_json(self, filename: str = "data/vrp_instance.json") -> None:
        """
        將生成的資料儲存為 JSON 檔案。
        
        Args:
            filename: 輸出檔案名稱
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        data = {
            "meta": {
                "map_size": self.map_size,
                "num_requests": self.num_requests
            },
            "fleet": self.fleet,
            "nodes": self.nodes
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"數據已生成並儲存至 {filename}")

    def visualize(self, output_file: str = "image/vrp_scenario.png") -> None:
        """
        視覺化生成的需求分布。
        
        Args:
            output_file: 輸出圖片檔案名稱
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 8))

        # 1. 為了讓Grid看起來不那麼像「完美的棋盤道路」，我們可以把透明度調低，
        #    或者只在特定區域畫Grid。這裡我們先把 grid 變淡，讓它看起來像「座標系統」而非「道路」。
        ax.grid(True, linestyle='--', alpha=0.2) 

        # 2. 繪製受限區域的「地形邊界」 (關鍵修改!)
        #    找出所有地形為 1 的點
        restricted_points = np.array([[n['x'], n['y']] for n in self.nodes if n['terrain'] == 1])
        
        if len(restricted_points) > 2: # 至少要3個點才能圍成一個面
            # 使用凸包 (Convex Hull) 算法自動圈出這些點
            hull = ConvexHull(restricted_points)
            # 取得頂點並封閉路徑
            vertices = restricted_points[hull.vertices]
            
            # 畫出一個淺紅色的多邊形區域，代表 "Restricted Terrain"
            poly = Polygon(vertices, facecolor='mistyrose', edgecolor='red', 
                           linestyle='--', alpha=0.4, label='Restricted Terrain')
            ax.add_patch(poly)
            
            # (選用) 在這個區域內加上文字標註
            cx = np.mean(restricted_points[:,0])
            cy = np.mean(restricted_points[:,1])
            ax.text(cx, cy, "UAV Only", color='darkred', fontsize=12, 
                    ha='center', va='center', fontweight='bold')
        
        # 繪製 Depot
        depot = self.nodes[0]
        plt.scatter(depot['x'], depot['y'], c='black', marker='s', s=100, label='Depot')
        
        # 分類節點以避免重複 label
        restricted_nodes = [n for n in self.nodes[1:] if n['terrain'] == 1]
        fast_nodes = [n for n in self.nodes[1:] if n['terrain'] == 0 and n['type'] == 'FAST']
        slow_nodes = [n for n in self.nodes[1:] if n['terrain'] == 0 and n['type'] == 'SLOW']
        
        # 繪製各類型節點
        if restricted_nodes:
            plt.scatter(
                [n['x'] for n in restricted_nodes],
                [n['y'] for n in restricted_nodes],
                c='red', marker='^', s=80, label='Restricted (UAV)'
            )
        if fast_nodes:
            plt.scatter(
                [n['x'] for n in fast_nodes],
                [n['y'] for n in fast_nodes],
                c='blue', marker='o', s=60, label='Normal Fast'
            )
        if slow_nodes:
            plt.scatter(
                [n['x'] for n in slow_nodes],
                [n['y'] for n in slow_nodes],
                c='green', marker='o', s=60, label='Normal Slow'
            )

        plt.legend()
        plt.title(f"Simulation Scenario (N={self.num_requests})")
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        plt.xlim(-5, self.map_size + 5)
        plt.ylim(-5, self.map_size + 5)
        plt.grid(True, alpha=0.3)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()  # 關閉圖形以釋放記憶體
        print(f"Visualization saved to {output_file}")

    def get_statistics(self) -> Dict[str, Any]:
        if not self.nodes:
            return {}
        
        customers = self.nodes[1:]
        
        # 修改點：這裡將分類切得更細，避免重疊
        restricted_fast = sum(1 for n in customers if n['terrain'] == 1)
        # 注意：這裡多加了 terrain == 0 的判斷，確保只算一般區域
        normal_fast = sum(1 for n in customers if n['terrain'] == 0 and n['type'] == 'FAST')
        normal_slow = sum(1 for n in customers if n['terrain'] == 0 and n['type'] == 'SLOW')
        
        return {
            "total_customers": len(customers),
            "restricted_fast": restricted_fast,
            "normal_fast": normal_fast,
            "normal_slow": normal_slow,
            # 增加一個驗證欄位，讓你知道快充總數是否正確
            "total_fast_sum": restricted_fast + normal_fast, 
            "total_demand": round(sum(n['demand'] for n in customers), 1),
            "avg_demand": round(sum(n['demand'] for n in customers) / len(customers), 1) if customers else 0
        }


# --- 執行生成 ---
if __name__ == "__main__":
    # 可指定 seed 以重現結果
    generator = VRPDataGenerator(map_size=100, num_requests=20, seed=45)
    generator.generate_requests()
    generator.generate_fleet()
    generator.save_to_json()
    generator.visualize()
    
    # 顯示統計資訊
    stats = generator.get_statistics()
    print(f"\n=== 生成統計 ===")
    print(f"總客戶數: {stats['total_customers']}")
    print(f"受限區域 (必為快充): {stats['restricted_fast']}")
    print(f"一般區域 (分配快充): {stats['normal_fast']}")
    print(f"一般區域 (分配慢充): {stats['normal_slow']}")
    print(f"--------------------------")
    print(f"快充總數 (受限+一般): {stats['total_fast_sum']}")