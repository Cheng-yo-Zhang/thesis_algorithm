import numpy as np
import copy
import random
from dataclasses import dataclass, field
from typing import List

# =====================================================================
# 1. 物理常數與資料結構
# =====================================================================
MCS_SPEED_M_S = 8.0             # MCS 移動速度 (約 28.8 km/h)
GRID_SIZE_M = 500.0             # 每格 500 公尺
TIME_PER_GRID_SEC = GRID_SIZE_M / MCS_SPEED_M_S  # 走一格約需 62.5 秒
MCS_MOVE_CONSUMPTION = 0.5      # kWh/grid
EV_CHARGE_DEMAND = 36.0         # 每次固定充 36 kWh
DEPOT_COORD = (0, 0)            # 基地座標 (用於回程充電)

@dataclass
class Task:
    task_id: int
    req_i: int
    req_j: int
    request_type: str        # 'Fast' or 'Slow'
    service_time_sec: float  # 1080s (18分) 或 5891s (98分)
    request_time_sec: float  # 發生時間 (相對 T=0 的秒數)
    actual_wait_time_sec: float = 0.0  # 【新增】真實等待時間
    actual_extra_delay_sec: float = 0.0  # 【新增】記錄行車與會合的總延遲
    max_tolerable_wait_sec: float = 0.0  # 【新增】該任務專屬的容忍等待上限 (秒)

@dataclass
class MCS:
    mcs_id: int
    current_i: int
    current_j: int
    current_battery: float = 270.0
    available_time_sec: float = 0.0  # 這台車「解除鎖定」可以接下一個任務的時間
    assigned_tasks: List[Task] = field(default_factory=list)

class ALNS_State:
    def __init__(self, fleet: List[MCS], unassigned: List[Task]):
        self.fleet = fleet
        self.unassigned = unassigned

    # 在 ALNS_State 類別中替換：
    def get_objective_value(self) -> float:
        """
        【真實的目標函數】：等待時間懲罰 + 行車延遲成本 + 未服務懲罰
        """
        total_system_cost = 0.0
        for mcs in self.fleet:
            for t in mcs.assigned_tasks:
                # IEEE 論文常見權重：原地等待的痛苦度 > 在路上開車的痛苦度
                total_system_cost += (t.actual_wait_time_sec * 2.0 + t.actual_extra_delay_sec * 1.0)
                
        penalty = len(self.unassigned) * 100000.0 
        return total_system_cost + penalty

# =====================================================================
# 升級版：核心約束與成本計算器 (支援 Multi-Trip 回程補充電力)
# =====================================================================
# =====================================================================
# 升級版：核心約束與成本計算器 (包含 Multi-Trip 回程與精準等待時間)
# =====================================================================
# =====================================================================
# 完美物理版：雙向奔赴與分離式延遲計算 (Meet-in-the-Middle)
# =====================================================================
# =====================================================================
# 完美物理版：雙向奔赴 (移除硬時間窗，觀測真實等待極限)
# =====================================================================
def evaluate_insertion(mcs: MCS, task: Task) -> dict:
    D_grids = abs(mcs.current_i - task.req_i) + abs(mcs.current_j - task.req_j)
    time_diff = mcs.available_time_sec - task.request_time_sec
    
    if time_diff <= 0:
        dist_ev_moves = D_grids / 2.0
        dist_mcs_moves = D_grids / 2.0
        ev_arrival_time = task.request_time_sec + (dist_ev_moves * TIME_PER_GRID_SEC)
        mcs_arrival_time = task.request_time_sec + (dist_mcs_moves * TIME_PER_GRID_SEC)
    else:
        ev_early_grids = time_diff / TIME_PER_GRID_SEC
        if ev_early_grids >= D_grids:
            dist_ev_moves = D_grids
            dist_mcs_moves = 0.0
            ev_arrival_time = task.request_time_sec + (dist_ev_moves * TIME_PER_GRID_SEC)
            mcs_arrival_time = mcs.available_time_sec
        else:
            remaining_grids = D_grids - ev_early_grids
            dist_ev_moves = ev_early_grids + (remaining_grids / 2.0)
            dist_mcs_moves = remaining_grids / 2.0
            meeting_time = mcs.available_time_sec + (dist_mcs_moves * TIME_PER_GRID_SEC)
            ev_arrival_time = meeting_time
            mcs_arrival_time = meeting_time

    # 計算等待時間 (僅計算原地罰站時間)
    if ev_arrival_time >= mcs_arrival_time:
        wait_time = 0.0  
    else:
        wait_time = mcs_arrival_time - ev_arrival_time  
        
    extra_delay = (dist_ev_moves * TIME_PER_GRID_SEC) + wait_time
    
    move_consumption = dist_mcs_moves * MCS_MOVE_CONSUMPTION
    dist_to_depot_from_meet = abs(task.req_i - DEPOT_COORD[0]) + abs(task.req_j - DEPOT_COORD[1])
    return_consumption = dist_to_depot_from_meet * MCS_MOVE_CONSUMPTION
    
    new_battery = mcs.current_battery - move_consumption - EV_CHARGE_DEMAND
    
    # ---------------------------------------------------------
    # 分支 1：電量不夠，MCS 先回基地充飽電再出發！
    # ---------------------------------------------------------
    if new_battery - return_consumption < 0:
        dist_to_depot = abs(mcs.current_i - DEPOT_COORD[0]) + abs(mcs.current_j - DEPOT_COORD[1])
        time_to_depot = dist_to_depot * TIME_PER_GRID_SEC
        
        recharge_time_sec = 1800.0  # 30 分鐘換電
        mcs_ready_at_depot_time = mcs.available_time_sec + time_to_depot + recharge_time_sec
        
        D_grids_depot = abs(DEPOT_COORD[0] - task.req_i) + abs(DEPOT_COORD[1] - task.req_j)
        time_diff_depot = mcs_ready_at_depot_time - task.request_time_sec
        
        if time_diff_depot <= 0:
            dist_ev_moves_depot = D_grids_depot / 2.0
            dist_mcs_moves_depot = D_grids_depot / 2.0
            ev_arr_depot = task.request_time_sec + (dist_ev_moves_depot * TIME_PER_GRID_SEC)
            mcs_arr_depot = task.request_time_sec + (dist_mcs_moves_depot * TIME_PER_GRID_SEC)
        else:
            ev_early_grids_depot = time_diff_depot / TIME_PER_GRID_SEC
            if ev_early_grids_depot >= D_grids_depot:
                dist_ev_moves_depot = D_grids_depot
                dist_mcs_moves_depot = 0.0
                ev_arr_depot = task.request_time_sec + (dist_ev_moves_depot * TIME_PER_GRID_SEC)
                mcs_arr_depot = mcs_ready_at_depot_time
            else:
                rem_grids = D_grids_depot - ev_early_grids_depot
                dist_ev_moves_depot = ev_early_grids_depot + (rem_grids / 2.0)
                dist_mcs_moves_depot = rem_grids / 2.0
                meeting_time_depot = mcs_ready_at_depot_time + (dist_mcs_moves_depot * TIME_PER_GRID_SEC)
                ev_arr_depot = meeting_time_depot
                mcs_arr_depot = meeting_time_depot
                
        if ev_arr_depot >= mcs_arr_depot:
            wait_time_depot = 0.0
        else:
            wait_time_depot = mcs_arr_depot - ev_arr_depot
            
        extra_delay_depot = (dist_ev_moves_depot * TIME_PER_GRID_SEC) + wait_time_depot
        consume_depot_to_meet = dist_mcs_moves_depot * MCS_MOVE_CONSUMPTION
        new_battery = 270.0 - consume_depot_to_meet - EV_CHARGE_DEMAND
        
        # 【新增：時間窗寬鬆度檢驗】
        if wait_time_depot > task.max_tolerable_wait_sec:
            return {'feasible': False, 'cost': float('inf')} # 超過該使用者的容忍極限，直接拒絕！
            
        consume_depot_to_meet = dist_mcs_moves_depot * MCS_MOVE_CONSUMPTION
        new_battery = 270.0 - consume_depot_to_meet - EV_CHARGE_DEMAND

        # 若滿電出發仍無法安全回程，則物理上不可行，拒絕
        if new_battery - return_consumption < 0:
            return {'feasible': False, 'cost': float('inf')}
            
        new_available_time = task.request_time_sec + extra_delay_depot + task.service_time_sec
        cost = extra_delay_depot * 1.0 + (dist_to_depot + dist_mcs_moves_depot) * MCS_MOVE_CONSUMPTION * 0.1
        
        return {
            'feasible': True,
            'cost': cost,
            'new_battery': new_battery,
            'new_time': new_available_time,
            'wait_time': wait_time_depot,
            'extra_delay': extra_delay_depot
        }

    # ---------------------------------------------------------
    # 分支 2：電量充足，直接過去會合
    # ---------------------------------------------------------
    if wait_time > task.max_tolerable_wait_sec:
        return {'feasible': False, 'cost': float('inf')} # 超過該使用者的容忍極限，直接拒絕！

    new_available_time = task.request_time_sec + extra_delay + task.service_time_sec
    cost = extra_delay * 1.0 + move_consumption * 0.1 

    return {
        'feasible': True,
        'cost': cost,
        'new_battery': new_battery,
        'new_time': new_available_time,
        'wait_time': wait_time,
        'extra_delay': extra_delay       # 【新增】
    }

# =====================================================================
# 新增：破壞算子 (Destroy Operator): Random Removal
# =====================================================================
def operator_random_removal(state: ALNS_State, num_to_remove: int = 5) -> ALNS_State:
    """隨機拔除任務，打破 ALNS 卡在區域最佳解的無限迴圈"""
    new_state = copy.deepcopy(state)
    removal_candidates = []
    
    # 將所有已分配的任務攤平，隨機抽取
    all_assigned = []
    for mcs in new_state.fleet:
        for t in mcs.assigned_tasks:
            all_assigned.append((mcs, t))
            
    if not all_assigned:
        return new_state
        
    # 隨機打亂並拔除
    random.shuffle(all_assigned)
    to_remove = all_assigned[:num_to_remove]
    
    for mcs, t in to_remove:
        mcs.assigned_tasks.remove(t)
        # 簡易還原電量 (實作上會有更嚴格的時間重置)
        mcs.current_battery = min(270.0, mcs.current_battery + EV_CHARGE_DEMAND)
        new_state.unassigned.append(t)
        
    return new_state

# =====================================================================
# 3. 破壞算子 (Destroy Operator): Worst Removal
# =====================================================================
def operator_worst_removal(state: ALNS_State, num_to_remove: int = 5) -> ALNS_State:
    """
    最差移除：找出車隊排程中，引起最大成本（例如繞極遠的路、等超久）的任務，將其拔除。
    """
    new_state = copy.deepcopy(state)
    removal_candidates = []

    for mcs in new_state.fleet:
        if not mcs.assigned_tasks:
            continue
            
        # 為了簡化範例，我們假設「最後一個被派發的任務」是最容易造成繞路的
        # 實作上會精算每個任務拔除後的 $\Delta Cost$
        worst_task = mcs.assigned_tasks.pop(-1)
        
        # 退回電量與時間 (近似還原)
        mcs.current_battery += EV_CHARGE_DEMAND
        
        new_state.unassigned.append(worst_task)
        removal_candidates.append(worst_task)
        
        if len(removal_candidates) >= num_to_remove:
            break
            
    return new_state

# =====================================================================
# 4. 修復算子 (Repair Operator): Regret-2 Insertion
# =====================================================================
# =====================================================================
# 4. 修復算子 (Repair Operator): Regret-2 Insertion (無新車版)
# =====================================================================
def operator_regret_insertion(state: ALNS_State) -> ALNS_State:
    new_state = copy.deepcopy(state)

    while new_state.unassigned:
        best_regret = -1
        best_task_idx = -1
        best_mcs_idx = -1
        best_insertion_info = None

        for t_idx, task in enumerate(new_state.unassigned):
            insertion_costs = []
            for m_idx, mcs in enumerate(new_state.fleet):
                # 正常評估，不再傳入 force_accept
                eval_result = evaluate_insertion(mcs, task)
                if eval_result['feasible']:
                    insertion_costs.append((eval_result['cost'], m_idx, eval_result))

            if not insertion_costs:
                continue

            insertion_costs.sort(key=lambda x: x[0])

            if len(insertion_costs) >= 2:
                regret_value = insertion_costs[1][0] - insertion_costs[0][0]
            elif len(insertion_costs) == 1:
                regret_value = float('inf')
            else:
                regret_value = -1

            if regret_value > best_regret:
                best_regret = regret_value
                best_task_idx = t_idx
                best_mcs_idx = insertion_costs[0][1]
                best_insertion_info = insertion_costs[0][2]

        # 如果所有現有車輛都因為電量物理極限無法接單，提早結束，留下未服務車輛
        if best_task_idx == -1:
            break

        # 正常的 Regret Insertion (安插給最佳現有車輛)
        target_task = new_state.unassigned.pop(best_task_idx)
        target_mcs = new_state.fleet[best_mcs_idx]
        
        target_task.actual_wait_time_sec = best_insertion_info['wait_time']
        target_task.actual_extra_delay_sec = best_insertion_info['extra_delay']
        target_mcs.assigned_tasks.append(target_task)
        target_mcs.current_battery = best_insertion_info['new_battery']
        target_mcs.available_time_sec = best_insertion_info['new_time']
        target_mcs.current_i = target_task.req_i
        target_mcs.current_j = target_task.req_j

    return new_state