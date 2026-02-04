"""Script to identify and remove ALNS-related code from main.py"""

with open('main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f'Total lines: {len(lines)}')

# Find key markers
for i, line in enumerate(lines):
    if 'class ALNSConfig' in line:
        print(f'ALNSConfig at line {i+1}')
    if 'class ALNSSolver:' in line:
        print(f'ALNSSolver at line {i+1}')
    if 'class ClusterALNSSolver:' in line:
        print(f'ClusterALNSSolver at line {i+1}')
    if 'def greedy_construction' in line:
        print(f'greedy_construction at line {i+1}')
    if 'def cluster_alns_construction' in line:
        print(f'cluster_alns_construction at line {i+1}')
    if '# ==================== 視覺化函式' in line:
        print(f'plot_routes section at line {i+1}')
    if '# ==================== 主程式類別' in line:
        print(f'ChargingSchedulingProblem at line {i+1}')
    if 'def main():' in line:
        print(f'main() at line {i+1}')
