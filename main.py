import random
import os
import json
import math
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


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
        """生成需求點數據，包含 Depot 和客戶節點"""
        # 先生成受限區域中心點
        self._generate_cluster_centers()
        
        # 生成 Depot（設在地圖中心）
        depot = {
            "id": 0,
            "type": "DEPOT",
            "x": 0,
            "y": 0,
            "demand": 0,
            "r_time": 0,
            "l_time": self.MINUTES_PER_DAY,
            "terrain": 0,
            "service_time": 0
        }
        self.nodes.append(depot)

        # 生成客戶需求
        for i in range(1, self.num_requests + 1):
            # 隨機座標
            x = random.randint(0, self.map_size)
            y = random.randint(0, self.map_size)
            
            # 判斷地形 (0: 一般, 1: 受限/UAV Only)
            terrain = 1 if self._is_in_restricted_area(x, y) else 0
            
            # 決定充電類型 (FAST / SLOW)
            # 受限區域強制為 FAST（假設山區救援都是急件）
            if terrain == 1:
                req_type = "FAST"
            else:
                req_type = "FAST" if random.random() < self.PROB_FAST else "SLOW"
            
            # 決定需求量 (kWh)
            if terrain == 1:  # UAV-only zones
                demand = round(random.uniform(4.0, self.UAV_MAX_PAYLOAD), 1)
            elif req_type == "FAST":
                demand = round(random.uniform(12, 40), 1)
            else:  # SLOW
                demand = round(random.uniform(20, 80), 1)
            
            # 計算服務時間
            service_time = self._calculate_service_time(demand, terrain, req_type)
            
            # 時間窗（反應式服務）
            request_time = random.randint(self.WORK_START_TIME, self.WORK_END_TIME)
            
            # 容忍等待時間（FAST/受限區域容忍度較低）
            if req_type == "FAST":
                tolerance = random.randint(30, 60)
            else:
                tolerance = random.randint(60, 180)
            deadline = request_time + tolerance

            node = {
                "id": i,
                "type": req_type,
                "x": x,
                "y": y,
                "demand": demand,
                "r_time": request_time,
                "l_time": deadline,
                "terrain": terrain,
                "service_time": service_time
            }
            self.nodes.append(node)

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
        plt.figure(figsize=(8, 8))
        
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
        """
        取得生成資料的統計資訊。
        
        Returns:
            包含各類統計數據的字典
        """
        if not self.nodes:
            return {}
        
        customers = self.nodes[1:]
        return {
            "total_customers": len(customers),
            "restricted_count": sum(1 for n in customers if n['terrain'] == 1),
            "fast_count": sum(1 for n in customers if n['type'] == 'FAST'),
            "slow_count": sum(1 for n in customers if n['type'] == 'SLOW'),
            "total_demand": round(sum(n['demand'] for n in customers), 1),
            "avg_demand": round(sum(n['demand'] for n in customers) / len(customers), 1) if customers else 0
        }


# --- 執行生成 ---
if __name__ == "__main__":
    # 可指定 seed 以重現結果
    generator = VRPDataGenerator(map_size=100, num_requests=20, seed=42)
    generator.generate_requests()
    generator.generate_fleet()
    generator.save_to_json()
    generator.visualize()
    
    # 顯示統計資訊
    stats = generator.get_statistics()
    print(f"\n=== 生成統計 ===")
    print(f"總客戶數: {stats['total_customers']}")
    print(f"受限區域: {stats['restricted_count']}")
    print(f"快充需求: {stats['fast_count']}")
    print(f"慢充需求: {stats['slow_count']}")
    print(f"總需求量: {stats['total_demand']} kWh")
    print(f"平均需求: {stats['avg_demand']} kWh")