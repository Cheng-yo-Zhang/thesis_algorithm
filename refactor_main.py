# Simple refactoring script for main.py
# Removes ALNS code, adds cluster-based construction

def main():
    # Read the original file
    with open('main.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Original lines: {len(lines)}")
    
    # Find section boundaries
    alns_config_start = -1
    alns_config_end = -1
    alns_solver_start = -1
    alns_solver_end = -1
    
    for i, line in enumerate(lines):
        if '@dataclass' in line and i+1 < len(lines) and 'class ALNSConfig' in lines[i+1]:
            alns_config_start = i
        if alns_config_start > 0 and alns_config_end < 0 and '# ==================== 節點類別' in line:
            alns_config_end = i
        if '# ==================== ALNS 求解器' in line:
            alns_solver_start = i
        if '# ==================== 視覺化函式' in line and alns_solver_start > 0:
            alns_solver_end = i
            break
    
    print(f"ALNSConfig: {alns_config_start} to {alns_config_end}")
    print(f"ALNSSolver: {alns_solver_start} to {alns_solver_end}")
    
    # Build new content
    new_lines = []
    
    # Add sklearn imports after line 9 (typing import)
    for i, line in enumerate(lines):
        new_lines.append(line)
        if 'from typing import' in line:
            new_lines.append('from sklearn.cluster import KMeans\n')
            new_lines.append('from sklearn.metrics import silhouette_score\n')
            new_lines.append('from sklearn.preprocessing import StandardScaler\n')
    
    # Now filter out ALNS sections
    filtered_lines = []
    skip = False
    
    for i, line in enumerate(new_lines):
        # Skip ALNSConfig
        if '@dataclass' in line and i+1 < len(new_lines) and 'class ALNSConfig' in new_lines[i+1]:
            skip = True
        if skip and '# ==================== 節點類別' in line:
            skip = False
        
        # Skip ALNSSolver  
        if '# ==================== ALNS 求解器' in line:
            skip = True
        if skip and '# ==================== 視覺化函式' in line:
            skip = False
            filtered_lines.append(line)
            continue
            
        if not skip:
            filtered_lines.append(line)
    
    print(f"New lines: {len(filtered_lines)}")
    
    # Write back
    with open('main.py', 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)
    
    print("Done!")

if __name__ == "__main__":
    main()
