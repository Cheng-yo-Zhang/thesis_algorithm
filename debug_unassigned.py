"""
Debug script to analyze unassigned nodes step by step
"""
import sys
sys.path.insert(0, '.')

from main import ChargingSchedulingProblem, Solution, Route

def main():
    problem = ChargingSchedulingProblem()
    problem.load_data('instance_25c_random_s42.csv')
    problem.assign_node_types(use_csv_types=True)
    
    # Run parallel insertion construction
    solution = problem.parallel_insertion_construction()
    
    print("\n" + "="*80)
    print("ANALYSIS OF UNASSIGNED NODES")
    print("="*80)
    
    if not solution.unassigned_nodes:
        print("All nodes were successfully assigned!")
        return
    
    for node in solution.unassigned_nodes:
        print(f"\n*** UNASSIGNED NODE: {node.id} ***")
        print(f"  Location: ({node.x}, {node.y})")
        print(f"  Type: {node.node_type}")
        print(f"  Demand: {node.demand} kWh")
        print(f"  Ready Time: {node.ready_time} min")
        print(f"  Due Date: {node.due_date} min")
        print(f"  Time Window Width: {node.due_date - node.ready_time} min")
        
        # Calculate travel time from depot
        travel_from_depot = problem.calculate_travel_time(problem.depot, node, 'mcs')
        print(f"\n  Distance from Depot: {problem.calculate_distance(problem.depot, node, 'manhattan'):.1f} km")
        print(f"  Travel Time from Depot (MCS): {travel_from_depot:.1f} min")
        
        # Calculate minimum service time
        if node.node_type == 'urgent':
            charging_power = problem.mcs.POWER_FAST  # 250 kW
            mode = "FAST (250 kW)"
        else:
            charging_power = problem.mcs.POWER_SLOW  # 11 kW
            mode = "SLOW (11 kW)"
        
        service_time = problem.calculate_charging_time(node.demand, charging_power)
        print(f"  Charging Mode: {mode}")
        print(f"  Service Time: {service_time:.1f} min")
        
        # Analyze feasibility
        print(f"\n  --- Feasibility Analysis (MCS from Depot) ---")
        
        # Dynamic dispatch: cannot depart before ready_time
        earliest_departure = node.ready_time
        earliest_arrival = earliest_departure + travel_from_depot
        earliest_completion = earliest_arrival + service_time
        
        print(f"  Earliest Departure from Depot: {earliest_departure:.1f} min (must wait for ready_time)")
        print(f"  Earliest Arrival at Node: {earliest_arrival:.1f} min")
        print(f"  Earliest Completion: {earliest_completion:.1f} min")
        print(f"  Due Date: {node.due_date:.1f} min")
        
        if earliest_arrival > node.due_date:
            print(f"\n  >> REASON: Cannot arrive before due date!")
            print(f"     Arrival ({earliest_arrival:.1f}) > Due Date ({node.due_date:.1f})")
        elif earliest_completion > node.due_date:
            print(f"\n  >> REASON: Cannot complete service before due date!")
            print(f"     Completion ({earliest_completion:.1f}) > Due Date ({node.due_date:.1f})")
            slack = node.due_date - earliest_arrival
            print(f"     Available slack after arrival: {slack:.1f} min")
            print(f"     Required service time: {service_time:.1f} min")
        else:
            print(f"\n  >> Node seems feasible from depot alone. Checking other constraints...")
        
        # Try UAV
        print(f"\n  --- UAV Feasibility Analysis ---")
        travel_uav = problem.calculate_travel_time(problem.depot, node, 'uav')
        uav_charging_power = problem.uav.CHARGE_POWER_KW  # 50 kW
        uav_service_time = problem.calculate_charging_time(node.demand, uav_charging_power)
        
        earliest_arrival_uav = node.ready_time + travel_uav
        earliest_completion_uav = earliest_arrival_uav + uav_service_time
        
        print(f"  UAV Travel Time from Depot: {travel_uav:.1f} min")
        print(f"  UAV Service Time (50 kW): {uav_service_time:.1f} min")
        print(f"  Earliest Arrival (UAV): {earliest_arrival_uav:.1f} min")
        print(f"  Earliest Completion (UAV): {earliest_completion_uav:.1f} min")
        
        # Check UAV capacity
        if node.demand > problem.uav.DELIVERABLE_ENERGY_KWH:
            print(f"\n  >> UAV INFEASIBLE: Demand ({node.demand} kWh) > UAV Capacity ({problem.uav.DELIVERABLE_ENERGY_KWH} kWh)")
        elif earliest_arrival_uav > node.due_date:
            print(f"\n  >> UAV INFEASIBLE: Cannot arrive before due date")
        elif earliest_completion_uav > node.due_date:
            print(f"\n  >> UAV INFEASIBLE: Cannot complete before due date")
        else:
            print(f"\n  >> UAV seems feasible - investigate further")

if __name__ == "__main__":
    main()
