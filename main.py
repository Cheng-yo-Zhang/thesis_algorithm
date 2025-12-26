import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
import matplotlib.pyplot as plt
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Set


# ==================== 參數配置 ====================
@dataclass
class MCSConfig:
    """Mobile Charging Station (MCS) 參數配置"""
    SPEED: float = (12 * 3.6) / 60.0   # km/min (原速度 12 m/s)
    CAPACITY: float = 270.0             # kWh 電池容量
    POWER_FAST: float = 250.0           # kW (Super Fast Charging)
    POWER_SLOW: float = 11.0            # kW (AC Slow Charging)


@dataclass
class UAVConfig:
    """Unmanned Aerial Vehicle (UAV) 參數配置"""
    SPEED: float = 60.0 / 60.0          # km/min (60 km/h)
    CAPACITY: float = 15.0              # kWh 電池容量
    POWER_FAST: float = 50.0            # kW 快充功率
    MAX_PAYLOAD: float = 12.0           # kWh 最大載電量


@dataclass
class ProblemConfig:
    """問題實例參數配置"""
    HARD_TO_ACCESS_RATIO: float = 0.10  # Hard-to-Access 節點比例
    URGENT_RATIO: float = 0.20          # Urgent 節點比例
    RANDOM_SEED: int = 42               # 隨機種子


# ==================== 節點類別 ====================
@dataclass
class Node:
    """節點資料結構"""
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float
    node_type: str = 'normal'  # 'depot', 'normal', 'hard_to_access', 'urgent'


# ==================== 主程式類別 ====================
class ChargingSchedulingProblem:
    """充電排程問題"""
    
    def __init__(self, 
                 mcs_config: MCSConfig = None,
                 uav_config: UAVConfig = None,
                 problem_config: ProblemConfig = None):
        
        self.mcs = mcs_config or MCSConfig()
        self.uav = uav_config or UAVConfig()
        self.config = problem_config or ProblemConfig()
        
        self.nodes: List[Node] = []
        self.depot: Node = None
        self.hard_to_access_indices: Set[int] = set()
        self.urgent_indices: Set[int] = set()
        
        # 設定隨機種子
        random.seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
    
    def load_data(self, filepath: str) -> None:
        """讀取 CSV 資料"""
        df = pd.read_csv(filepath)
        
        for idx, row in df.iterrows():
            node = Node(
                id=int(row['CUST NO.']),
                x=float(row['XCOORD.']),
                y=float(row['YCOORD.']),
                demand=float(row['DEMAND']),
                ready_time=float(row['READY TIME']),
                due_date=float(row['DUE DATE']),
                service_time=float(row['SERVICE TIME'])
            )
            
            if idx == 0:
                node.node_type = 'depot'
                self.depot = node
            
            self.nodes.append(node)
        
        print(f"載入 {len(self.nodes)} 個節點 (含 depot)")
    
    def assign_node_types(self) -> None:
        """隨機分配節點類型"""
        customer_indices = [i for i in range(1, len(self.nodes))]
        num_customers = len(customer_indices)
        
        # 隨機選取 Hard-to-Access 節點
        num_hard = int(num_customers * self.config.HARD_TO_ACCESS_RATIO)
        self.hard_to_access_indices = set(random.sample(customer_indices, num_hard))
        
        for idx in self.hard_to_access_indices:
            self.nodes[idx].node_type = 'hard_to_access'
        
        # 從剩餘節點選取 Urgent 節點
        remaining = [i for i in customer_indices if i not in self.hard_to_access_indices]
        num_urgent = int(num_customers * self.config.URGENT_RATIO)
        self.urgent_indices = set(random.sample(remaining, num_urgent))
        
        for idx in self.urgent_indices:
            self.nodes[idx].node_type = 'urgent'
        
        print(f"Hard-to-Access 節點: {len(self.hard_to_access_indices)} 個")
        print(f"Urgent 節點: {len(self.urgent_indices)} 個")
        print(f"Normal 節點: {num_customers - len(self.hard_to_access_indices) - len(self.urgent_indices)} 個")
    
    def calculate_distance(self, node1: Node, node2: Node, distance_type: str = 'euclidean') -> float:
        """
        計算兩節點間的距離
        
        Args:
            node1: 起點節點
            node2: 終點節點
            distance_type: 'euclidean' (歐幾里得) 或 'manhattan' (曼哈頓)
        
        Returns:
            距離值
        """
        if distance_type == 'manhattan':
            # 曼哈頓距離 (適用於地面車輛 MCS)
            return abs(node1.x - node2.x) + abs(node1.y - node2.y)
        else:
            # 歐幾里得距離 (適用於 UAV 直線飛行)
            return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def calculate_travel_time(self, node1: Node, node2: Node, vehicle: str = 'mcs') -> float:
        """
        計算行駛時間 (分鐘)
        
        Args:
            node1: 起點節點
            node2: 終點節點
            vehicle: 'mcs' (地面車) 或 'uav' (無人機)
        
        Returns:
            行駛時間 (分鐘)
        """
        if vehicle == 'mcs':
            # MCS 使用曼哈頓距離
            distance = self.calculate_distance(node1, node2, distance_type='manhattan')
            return distance / self.mcs.SPEED
        else:
            # UAV 使用歐幾里得距離
            distance = self.calculate_distance(node1, node2, distance_type='euclidean')
            return distance / self.uav.SPEED
    
    def calculate_charging_time(self, energy_kwh: float, power_kw: float) -> float:
        """計算充電時間 (分鐘)"""
        return (energy_kwh / power_kw) * 60.0
    
    def plot_nodes(self, save_path: str = 'node_distribution.png') -> None:
        """繪製節點分布圖"""
        plt.figure(figsize=(10, 8))
        
        # 分類節點
        normal_nodes = [n for n in self.nodes if n.node_type == 'normal']
        hard_nodes = [n for n in self.nodes if n.node_type == 'hard_to_access']
        urgent_nodes = [n for n in self.nodes if n.node_type == 'urgent']
        
        # 繪製普通節點 (藍色)
        if normal_nodes:
            plt.scatter([n.x for n in normal_nodes], [n.y for n in normal_nodes],
                       c='blue', marker='o', s=50, label='Normal')
        
        # 繪製 Hard-to-Access 節點 (紅色)
        if hard_nodes:
            plt.scatter([n.x for n in hard_nodes], [n.y for n in hard_nodes],
                       c='red', marker='^', s=80, label='Hard-to-Access (10%)')
        
        # 繪製 Urgent 節點 (橘色)
        if urgent_nodes:
            plt.scatter([n.x for n in urgent_nodes], [n.y for n in urgent_nodes],
                       c='orange', marker='s', s=80, label='Urgent (20%)')
        
        # 繪製 Depot (綠色)
        if self.depot:
            plt.scatter(self.depot.x, self.depot.y,
                       c='green', marker='*', s=200, label='Depot')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('R101 Node Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"圖片已儲存至: {save_path}")
    
    def print_config(self) -> None:
        """輸出參數配置"""
        print("\n" + "="*50)
        print("MCS 參數配置:")
        print(f"  速度: {self.mcs.SPEED:.3f} km/min ({self.mcs.SPEED * 60:.1f} km/h)")
        print(f"  電池容量: {self.mcs.CAPACITY} kWh")
        print(f"  快充功率: {self.mcs.POWER_FAST} kW")
        print(f"  慢充功率: {self.mcs.POWER_SLOW} kW")
        print("\nUAV 參數配置:")
        print(f"  速度: {self.uav.SPEED:.3f} km/min ({self.uav.SPEED * 60:.1f} km/h)")
        print(f"  電池容量: {self.uav.CAPACITY} kWh")
        print(f"  快充功率: {self.uav.POWER_FAST} kW")
        print(f"  最大載電量: {self.uav.MAX_PAYLOAD} kWh")
        print("="*50 + "\n")


# ==================== 主程式入口 ====================
def main():
    # 初始化問題
    problem = ChargingSchedulingProblem()
    
    # 輸出參數配置
    problem.print_config()
    
    # 載入資料
    problem.load_data('R101.csv')
    
    # 分配節點類型
    problem.assign_node_types()
    
    # 繪製節點分布圖
    problem.plot_nodes()
    
    # 輸出節點資訊
    print(f"\nHard-to-Access 節點索引: {sorted(problem.hard_to_access_indices)}")
    print(f"Urgent 節點索引: {sorted(problem.urgent_indices)}")


if __name__ == "__main__":
    main()

