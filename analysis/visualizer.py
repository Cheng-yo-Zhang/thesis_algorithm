# analysis/visualizer.py
import matplotlib
matplotlib.use('Agg') # 避免 TclError
import matplotlib.pyplot as plt
from config import settings

def plot_scenario_snapshot(requests, chargers, time_slot, traffic, filename):
    plt.figure(figsize=(10, 10))
    plt.xlim(0, settings.GRID_SIZE)
    plt.ylim(0, settings.GRID_SIZE)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 1. 畫熱區圈圈 (背景)
    ax = plt.gca()
    for z in settings.GENERATION_HOT_ZONES:
        circle = plt.Circle(z['center'], z['sigma']*2, color='gray', alpha=0.1)
        ax.add_patch(circle)

    # 2. 畫車隊位置 (驗證動態偏遠)
    for c in chargers:
        if 'UAV' in c.type:
            plt.scatter(c.x, c.y, marker='^', c='red', s=120, edgecolors='k', label='UAV')
        elif 'MCS' in c.type:
            plt.scatter(c.x, c.y, marker='s', c='blue', s=120, edgecolors='k', label='MCS')

    # 3. 畫需求點
    colors = {'URGENT_UAV': 'red', 'FAST_MCS': 'orange', 'SLOW_MCS': 'green'}
    for r in requests:
        plt.scatter(r.x, r.y, c=colors.get(r.req_type, 'black'), s=50, alpha=0.7)

    # 避免 Legend 重複
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # 手動補上 Request 的圖例
    from matplotlib.lines import Line2D
    by_label['Req: UAV'] = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8)
    by_label['Req: Fast'] = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8)
    by_label['Req: Slow'] = Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8)
    
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.title(f"Time: {time_slot} | Traffic: {traffic:.2f}\nFleet Logic: MCS Dist > {settings.REMOTE_DISTANCE_THRESHOLD} -> Remote")
    plt.xlabel("X (Grid)")
    plt.ylabel("Y (Grid)")
    
    plt.savefig(filename)
    plt.close()