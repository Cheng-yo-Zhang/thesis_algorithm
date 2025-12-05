# main.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import settings
from core.models import Charger
from simulation.generator import RequestGenerator
from analysis.visualizer import plot_scenario_snapshot
from algorithms.clustering import ClusterModel

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def initialize_chargers():
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
    return chargers

def process_fleet_movement(chargers):
    """
    處理所有充電車的移動邏輯 (每個 Time Slot 呼叫一次)
    """
    for c in chargers:
        if c.status == 'MOVING' and c.target_x is not None:
            # 1. 計算距離
            dist_meters = c.speed * settings.TIME_SLOT_DURATION
            dist_grid = dist_meters / (settings.ROAD_LENGTH * 1000)
            
            dx = c.target_x - c.x
            dy = c.target_y - c.y
            dist_to_target = np.sqrt(dx**2 + dy**2)
            
            # 2. 移動與狀態更新
            if dist_to_target <= dist_grid:
                # 到達目的地
                c.x = c.target_x
                c.y = c.target_y
                c.clear_target() # 變回 IDLE，等待服務或下一次指派
                
                # [模擬] 到達後扣除一點移動耗電 (簡化)
                c.current_energy -= 0.5 
            else:
                # 移動一步
                ratio = dist_grid / dist_to_target
                c.x += dx * ratio
                c.y += dy * ratio
                # 移動耗電
                c.current_energy -= 0.1 

            # 邊界與電量檢查
            c.x = np.clip(c.x, 0, settings.GRID_SIZE)
            c.y = np.clip(c.y, 0, settings.GRID_SIZE)
            if c.current_energy < 0: c.current_energy = 0

def dispatch_logic(current_time, chargers, pending_requests, cluster_model):
    """
    核心調度邏輯：
    1. 篩選 IDLE 車輛
    2. 篩選緊急需求
    3. 分群與指派
    """
    
    # A. 找出閒置且有電的 MCS (UAV 暫不在此邏輯)
    idle_mcs = [
        c for c in chargers 
        if c.type == 'MCS' and c.status == 'IDLE' and c.current_energy > 50 # 電量閾值
    ]
    
    if not idle_mcs or not pending_requests:
        return # 沒車或沒單，收工

    # B. 需求處理 (Queueing)
    # 這裡可以加入時間窗過濾：只處理快要到期的 (DueTime - CurrentTime < Threshold)
    # 目前先簡單處理：處理所有 Pending
    target_requests = pending_requests 

    # C. 決定分群數量 K (Demand-Driven)
    # K 不能超過閒置車輛數，也不能超過需求數
    # 策略：有多少閒置車，就盡量服務多少群 (或是需求少時，就只分需求數量的群)
    n_clusters = min(len(idle_mcs), len(target_requests))
    
    # D. 計算重心
    centroids = cluster_model.get_cluster_centroids(target_requests, n_clusters)
    
    # E. 媒合 (Matching) - 簡單貪婪法
    # 將每個重心指派給最近的 IDLE MCS
    assigned_indices = set()
    
    for cx, cy in centroids:
        best_mcs = None
        min_dist = float('inf')
        
        # 找最近的車
        for i, mcs in enumerate(idle_mcs):
            if i in assigned_indices: continue
            
            dist = abs(mcs.x - cx) + abs(mcs.y - cy)
            if dist < min_dist:
                min_dist = dist
                best_mcs = mcs
                best_idx = i
        
        # 指派任務
        if best_mcs:
            best_mcs.set_target(cx, cy)
            assigned_indices.add(best_idx)
            # print(f"[Dispatch] Time {current_time}: MCS {best_mcs.id} -> ({cx:.1f}, {cy:.1f})")

def main():
    print("=== Simulation Started (Refined Logic) ===")
    
    chargers = initialize_chargers()
    generator = RequestGenerator()
    cluster_model = ClusterModel()
    
    # [新增] 全域待處理清單
    all_pending_requests = [] 

    for t in range(settings.TOTAL_TIME_SLOTS):
        
        # 1. 生成新需求 (Generate)
        # 這裡要注意 generator 回傳的是"這一刻新增的"，不是所有的
        new_requests, traffic = generator.generate_requests(t, chargers)
        
        # 為新需求加上時間標記 (如果 generator 沒加，這裡補)
        for r in new_requests:
            r.due_time = t + r.max_tolerance # 設定最後期限
            all_pending_requests.append(r)
            
        # 2. 清理過期或已完成的需求 (Clean up)
        # 這裡簡化：假設只要 MCS 到達附近，需求就被視為「將被服務」，從 pending 移除
        # 實際邏輯應該更複雜 (MCS status = SERVING)，但為了讓調度流暢，我們先這樣做
        valid_requests = []
        for r in all_pending_requests:
            # 簡單判定：如果有 MCS 在它附近 (例如 1km 內)，算已服務
            is_covered = any(
                c for c in chargers 
                if c.type == 'MCS' and np.sqrt((c.x-r.x)**2 + (c.y-r.y)**2) < 2.0
            )
            if not is_covered and t < r.due_time:
                valid_requests.append(r)
        
        all_pending_requests = valid_requests

        # 3. 執行調度 (Dispatch) - 針對還在 pending 的需求
        # 設定頻率：例如每 10 分鐘調度一次，或是隨時調度
        if t % 10 == 0: 
            dispatch_logic(t, chargers, all_pending_requests, cluster_model)
        
        # 4. 物理移動 (Move)
        process_fleet_movement(chargers)
        
        # 5. 輸出快照 (Snapshot)
        if t % 60 == 0:
            print(f"Time {t:04d} | Pending Reqs: {len(all_pending_requests)} | Idle MCS: {len([c for c in chargers if c.type=='MCS' and c.status=='IDLE'])}")
            
            # 繪圖時傳入 all_pending_requests 才能看到累積的需求
            filename = os.path.join(OUTPUT_DIR, f"snapshot_t{t:04d}.png")
            plot_scenario_snapshot(all_pending_requests, chargers, t, traffic, filename)

    print("=== Simulation Finished ===")

if __name__ == "__main__":
    main()