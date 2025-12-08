# algorithms/clustering.py
import numpy as np
from config import settings

class ClusterModel:
    def __init__(self):
        self.mcs_capacity = settings.CHARGER_CONFIG['MCS']['capacity']
        self.remote_threshold = settings.REMOTE_DISTANCE_THRESHOLD # 引用設定檔的 40

    def get_capacitated_clusters(self, requests):
        """
        輸入: requests (List[Request])
        輸出: clusters_info (List[dict])
        邏輯: 貪婪容量分群 + 重心距離過濾
        """
        # 只處理還不是 UAV 的單 (避免重複處理)
        pending = [r for r in requests if 'UAV' not in str(r.req_type)]
        clusters_info = []

        while pending:
            # 1. [選種子] 簡單策略：選清單中第一個 (通常是最早生成的)
            seed_req = pending.pop(0)
            
            current_cluster_reqs = [seed_req]
            current_energy = seed_req.energy_demand
            
            # 2. [容量吸納] 嘗試把附近的點吸進來
            if pending:
                # 計算 pending 中所有點到 seed 的距離
                dists = [
                    (r, (r.x - seed_req.x)**2 + (r.y - seed_req.y)**2) 
                    for r in pending
                ]
                dists.sort(key=lambda x: x[1]) # 由近到遠
                
                remaining_after_pass = []
                for r, _ in dists:
                    if current_energy + r.energy_demand <= self.mcs_capacity:
                        current_cluster_reqs.append(r)
                        current_energy += r.energy_demand
                    else:
                        remaining_after_pass.append(r)
                
                # 更新 pending (剩下沒被吸走的)
                pending = remaining_after_pass

            # 3. [計算暫定重心]
            cx = np.mean([r.x for r in current_cluster_reqs])
            cy = np.mean([r.y for r in current_cluster_reqs])
            
            # 4. [距離剔除] 檢查是否有人離重心太遠
            final_cluster_reqs = []
            final_energy = 0.0
            
            for r in current_cluster_reqs:
                dist_to_centroid = abs(r.x - cx) + abs(r.y - cy) # 曼哈頓距離
                
                if dist_to_centroid <= self.remote_threshold:
                    final_cluster_reqs.append(r)
                    final_energy += r.energy_demand
                else:
                    # === 關鍵修改 ===
                    # 離重心太遠 -> 視為偏遠地區 -> 轉為 UAV 類型
                    r.req_type = 'URGENT_UAV'
                    r.note = f"Too far from cluster centroid (Dist: {dist_to_centroid:.1f})"
                    # 注意：這裡不把它加回 pending，因為它已經變成 UAV 單了，
                    # 下一輪 MCS 分群時會直接被第一行的 filter 擋掉。
            
            # 5. [存檔] 如果剔除後還有剩，才算成一團
            if final_cluster_reqs:
                # 選擇性：是否要根據剔除後的成員重新計算重心？
                # 為了精確，我們重新算一次
                new_cx = np.mean([r.x for r in final_cluster_reqs])
                new_cy = np.mean([r.y for r in final_cluster_reqs])
                
                clusters_info.append({
                    'centroid': (new_cx, new_cy),
                    'requests': final_cluster_reqs,
                    'total_energy': final_energy
                })

        return clusters_info