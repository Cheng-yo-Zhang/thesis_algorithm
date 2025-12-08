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
        if c.status == 'SERVING':
            c.service_timer -= 1
            power_used = c.current_charging_power
            energy_delivered = power_used * (settings.TIME_SLOT_DURATION / 3600.0)
            c.current_energy -= energy_delivered
            if c.service_timer <= 0 or c.current_energy <= 0:
                for r in c.assigned_requests:
                    r.status = 'COMPLETED'
                c.assigned_requests = [] # 清空手上的單

                c.status = 'IDLE'
                c.service_timer = 0
                c.current_charging_power = 0.0 # 歸零
            continue
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

                if c.next_service_duration > 0:
                    c.status = 'SERVING'  # 切換狀態！
                    c.service_timer = c.next_service_duration # 設定計時器
                    c.next_service_duration = 0 # 重置緩衝

                    for r in c.assigned_requests:
                        r.status = 'SERVING'
                    
                    c.target_x = None
                    c.target_y = None
                else:
                    c.clear_target()
                
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
    # 篩選還沒被服務的單
    valid_pending = [r for r in pending_requests if r.status == 'PENDING']
    
    # 篩選閒置 MCS
    idle_mcs = [c for c in chargers if c.type == 'MCS' and c.status == 'IDLE' and c.current_energy > 50]
    
    if not idle_mcs or not valid_pending:
        return []

    # 1. 取得分群 (演算法內部已經處理了容量與偏遠判定)
    clusters_info = cluster_model.get_capacitated_clusters(valid_pending)
    
    centroids_for_plot = []
    
    # 2. 指派
    num_to_assign = min(len(idle_mcs), len(clusters_info))
    
    for i in range(num_to_assign):
        cluster = clusters_info[i]
        mcs = idle_mcs[i]
        
        cx, cy = cluster['centroid']
        reqs = cluster['requests']
        total_energy = cluster['total_energy']
        
        # 設定 MCS
        mcs.set_target(cx, cy)
        mcs.assigned_requests = reqs
        
        # 標記 Request
        for r in reqs:
            r.status = 'ASSIGNED'
        
        # 估算時間
        charging_power = settings.MCS_POWER_MAX
        required_hours = total_energy / charging_power
        mcs.next_service_duration = int(np.ceil(required_hours * 60)) + 10 # +10分鐘緩衝
        mcs.current_charging_power = charging_power
        
        centroids_for_plot.append((cx, cy))

    return centroids_for_plot

def main():
    print("=== Simulation Started (Capacitated Clustering Mode) ===")
    
    chargers = initialize_chargers()
    generator = RequestGenerator()
    cluster_model = ClusterModel()
    
    current_sim_time = 0 
    
    for traffic_slot in range(settings.TOTAL_TIME_SLOTS):
        
        if traffic_slot == 0:
            continue
            
        # 1. 生成需求
        new_requests, traffic = generator.generate_requests(traffic_slot, chargers)
        if not new_requests:
            continue
            
        print(f"\n[Batch {traffic_slot}] Incoming Reqs: {len(new_requests)}")
        batch_requests = new_requests
        # 初始化狀態
        for r in batch_requests: 
            r.status = 'PENDING'
            r.due_time = 999999 # Batch 模式下忽略 deadline

        # 2. 進入服務迴圈 (直到這批做完)
        # 注意：因為可能車少單多，我們可能要在迴圈內多次分群
        
        plot_done = False # 控制只畫第一張規劃圖

        while any(r.status != 'COMPLETED' for r in batch_requests):

            for r in batch_requests:
                if r.status == 'PENDING' and r.req_type == 'URGENT_UAV':
                    r.status = 'COMPLETED' # 或標記為 'HANDOVER' 視同離開此系統
                    print(f"   -> [System] Req {r.id} is Remote (UAV). Handover & Close.")
            
            # (A) 檢查是否需要調度
            # 條件：有 PENDING 的單 且 有 IDLE 的車
            pending_count = sum(1 for r in batch_requests if r.status == 'PENDING')
            idle_mcs_count = sum(1 for c in chargers if c.type == 'MCS' and c.status == 'IDLE')
            
            if pending_count > 0 and idle_mcs_count > 0:
                # 再次分群並派車
                centroids = dispatch_logic(current_sim_time, chargers, batch_requests, cluster_model)
                
                # 繪圖：只在每個 Batch 第一次派車時畫，或者每次派車都畫
                # 這裡設定：每個 Batch 的第一次規劃畫一張圖
                if not plot_done and centroids:
                    filename = os.path.join(OUTPUT_DIR, f"plan_batch_{traffic_slot:04d}.png")
                    plot_scenario_snapshot(batch_requests, chargers, current_sim_time, traffic, filename, centroids=centroids)
                    print(f"   -> Plan plotted: {filename}")
                    plot_done = True

            # (B) 物理移動與服務
            process_fleet_movement(chargers)
            current_sim_time += 1

        print(f"[Batch {traffic_slot}] All Cleared at Sim Time {current_sim_time}")

    print("=== Simulation Finished ===")

if __name__ == "__main__":
    main()