# analysis/visualizer.py
import matplotlib
matplotlib.use('Agg') # 避免 TclError
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # 記得引入這個
from config import settings

# [修改] 這裡新增了 centroids=None 參數
def plot_scenario_snapshot(requests, chargers, time_slot, traffic, filename, centroids=None):
    plt.figure(figsize=(10, 10))
    plt.xlim(0, settings.GRID_SIZE)
    plt.ylim(0, settings.GRID_SIZE)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 1. 畫分群重心 (Centroids) - [新增功能]
    # 這代表演算法計算出的「最佳服務位置」
    if centroids:
        # 解壓縮 (x, y)
        cx_list = [c[0] for c in centroids]
        cy_list = [c[1] for c in centroids]
        plt.scatter(cx_list, cy_list, marker='x', c='black', s=200, linewidths=3, zorder=10, label='Centroid')

    # 2. 畫車隊位置
    for c in chargers:
        if 'UAV' in c.type:
            plt.scatter(c.x, c.y, marker='^', c='red', s=120, edgecolors='k', label='UAV', zorder=5)
        elif 'MCS' in c.type:
            # 區分狀態顏色：服務中(Cyan) / 閒置(Blue)
            color = 'cyan' if c.status == 'SERVING' else 'blue'
            plt.scatter(c.x, c.y, marker='s', c=color, s=120, edgecolors='k', label='MCS', zorder=5)

    # 3. 畫需求點
    colors = {'URGENT_UAV': 'red', 'FAST_MCS': 'orange', 'SLOW_MCS': 'green'}
    for r in requests:
        # 完成的單畫淡一點，UAV 單畫紅色
        c = colors.get(r.req_type, 'black')
        if r.status == 'COMPLETED':
            c = 'gray'
        
        alpha = 0.3 if r.status == 'COMPLETED' else 0.8
        plt.scatter(r.x, r.y, c=c, s=50, alpha=alpha, zorder=3)

    # 4. 手動設定圖例 (避免重複與遺漏)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # 補上 Request 的圖例
    by_label['Req: Fast'] = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8)
    by_label['Req: Slow'] = Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8)
    by_label['Req: UAV']  = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8)
    
    if centroids:
        by_label['Centroid'] = Line2D([0], [0], marker='x', color='w', markeredgecolor='black', markersize=10, markeredgewidth=2)
    
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.title(f"Time: {time_slot} | Traffic: {traffic:.2f}\nMode: Stop-and-Go")
    plt.xlabel("X (Grid)")
    plt.ylabel("Y (Grid)")
    
    plt.savefig(filename)
    plt.close()