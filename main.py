import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # 確保後端正確，避免跳出視窗錯誤
import matplotlib.pyplot as plt

# 引入模組
from config import settings
from core.models import Charger
from simulation.generator import RequestGenerator
from analysis.visualizer import plot_scenario_snapshot
# [新增] 引入分群演算法
from algorithms.clustering import ClusterModel

# 建立結果輸出資料夾
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def initialize_chargers():
    """
    初始化所有充電資源，初始位置都在 DEPOT_LOCATION (0, 0)
    """
    chargers = []
    c_id = 0
    for type_name, cfg in settings.CHARGER_CONFIG.items():
        count = cfg['count']
        for _ in range(count):
            new_charger = Charger(
                charger_id=c_id,
                start_x=settings.DEPOT_LOCATION[0],
                start_y=settings.DEPOT_LOCATION[1],
                type_name=type_name
            )
            chargers.append(new_charger)
            c_id += 1
    print(f"[Init] Total Chargers Created: {len(chargers)}")
    return chargers

def move_charger_towards_target(charger, target_x, target_y):
    """
    物理移動邏輯：
    根據 Charger 的速度 (speed m/s) 與時間槽長度 (60s)，
    計算它在網格上能移動多少距離，並更新 (x, y)。
    """
    # 1. 計算該時間槽能移動的真實距離 (公尺)
    # distance (m) = speed (m/s) * time (s)
    dist_meters = charger.speed * settings.TIME_SLOT_DURATION
    
    # 2. 換算成網格距離 (Grid Units)
    # 1 Grid = ROAD_LENGTH km = ROAD_LENGTH * 1000 meters
    dist_grid = dist_meters / (settings.ROAD_LENGTH * 1000)
    
    # 3. 計算當前與目標的距離 (曼哈頓或歐幾里得皆可，移動本身通常走直線向量逼近)
    dx = target_x - charger.x
    dy = target_y - charger.y
    current_dist_to_target = np.sqrt(dx**2 + dy**2)
    
    # 4. 更新座標
    if current_dist_to_target <= dist_grid:
        # 如果距離夠近，直接到達
        charger.x = target_x
        charger.y = target_y
    else:
        # 否則，依照比例移動一步
        ratio = dist_grid / current_dist_to_target
        charger.x += dx * ratio
        charger.y += dy * ratio
        
    # 確保不超出邊界
    charger.x = np.clip(charger.x, 0, settings.GRID_SIZE)
    charger.y = np.clip(charger.y, 0, settings.GRID_SIZE)

def dispatch_and_move_mcs(chargers, requests, cluster_model):
    """
    核心調度與移動邏輯：
    1. 篩選需求 (Fast / Slow)
    2. 使用 K-Means 計算重心 (Targets)
    3. 指揮 MCS 移動到重心
    """
    
    # --- A. 準備資料 ---
    # 篩選出 Fast 和 Slow 的需求點
    fast_reqs = [r for r in requests if r.req_type == 'FAST_MCS']
    slow_reqs = [r for r in requests if r.req_type == 'SLOW_MCS']
    
    # 篩選出對應的 MCS 車隊
    fast_mcs_fleet = [c for c in chargers if c.type == 'FastMCS']
    slow_mcs_fleet = [c for c in chargers if c.type == 'SlowMCS']
    
    # --- B. 計算 FastMCS 的目標重心 ---
    if fast_mcs_fleet:
        # 計算 K 個重心 (K = 車隊數量，讓每台車都有地方去)
        # 如果需求很少，重心數量會自動縮減 (在 ClusterModel 裡處理)
        targets = cluster_model.get_cluster_centroids(fast_reqs, len(fast_mcs_fleet))
        
        # 簡單指派：第 i 台車去第 i 個重心 (若重心不夠，就去最後一個)
        for i, mcs in enumerate(fast_mcs_fleet):
            if targets:
                # 取模運算確保不會 index out of range (當重心數 < 車數時)
                target = targets[i % len(targets)]
                move_charger_towards_target(mcs, target[0], target[1])
            # 若完全無需求 (targets為空)，MCS 留在原地或可設定返回 Depot

    # --- C. 計算 SlowMCS 的目標重心 ---
    if slow_mcs_fleet:
        targets = cluster_model.get_cluster_centroids(slow_reqs, len(slow_mcs_fleet))
        for i, mcs in enumerate(slow_mcs_fleet):
            if targets:
                target = targets[i % len(targets)]
                move_charger_towards_target(mcs, target[0], target[1])

def main():
    print("=== Simulation Started ===")
    
    # 1. 初始化物件
    chargers = initialize_chargers()
    generator = RequestGenerator()
    cluster_model = ClusterModel() # 分群演算法
    
    # 設定快照間隔 (每 60 分鐘拍一次)
    snapshot_interval = 60 
    
    # 2. 時間迴圈
    for t in range(settings.TOTAL_TIME_SLOTS):
        
        # A. 生成需求
        # 注意：這裡傳入的是 "移動前" 的 chargers 位置來計算偏遠度
        # 這符合邏輯：上一刻的車隊位置決定了這一刻哪些地方算偏遠
        requests, traffic = generator.generate_requests(t, chargers)
        
        # B. 調度與移動 (這是讓 MCS 散開的關鍵)
        # 根據剛剛生成的需求，計算重心並移動 MCS
        dispatch_and_move_mcs(chargers, requests, cluster_model)
        
        # C. 輸出 Log 與 快照
        if t % 60 == 0:
            uav_reqs = len([r for r in requests if r.req_type == 'URGENT_UAV'])
            fast_reqs = len([r for r in requests if r.req_type == 'FAST_MCS'])
            slow_reqs = len([r for r in requests if r.req_type == 'SLOW_MCS'])
            
            print(f"Time {t:04d} | Traffic: {traffic:.2f} | Req: {len(requests)} (UAV:{uav_reqs}, Fast:{fast_reqs}, Slow:{slow_reqs})")
            
            filename = os.path.join(OUTPUT_DIR, f"snapshot_t{t:04d}.png")
            plot_scenario_snapshot(requests, chargers, t, traffic, filename)
            print(f"   >>> Snapshot saved: {filename}")

    print("=== Simulation Finished ===")

if __name__ == "__main__":
    main()