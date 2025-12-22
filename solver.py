"""
Genetic Algorithm Solver for Dual-Mode Mobile Charging VRP with UAVs.

This solver implements a Route-First, Cluster-Second approach using:
- Two separate gene sequences (MCS and UAV)
- Greedy split decoder for route construction
- Order Crossover (OX) and Swap Mutation
- Tournament Selection

Author: GA Solver for Air-Ground Collaborative Scheduling
"""

import json
import random
import math
import copy
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class GASolver:
    """
    Genetic Algorithm Solver for the Mobile Charging VRP with UAV collaboration.
    
    Attributes:
        data: Problem instance data loaded from JSON
        nodes: List of all nodes (depot + customers)
        fleet: Fleet configuration for MCS and UAV
        distance_matrix: Precomputed distance matrix
        time_matrix_mcs: Travel time matrix for MCS
        time_matrix_uav: Travel time matrix for UAV
    """
    
    # GA Parameters
    POPULATION_SIZE = 100
    MAX_GENERATIONS = 500
    CROSSOVER_RATE = 0.85
    MUTATION_RATE = 0.15
    TOURNAMENT_SIZE = 5
    ELITE_SIZE = 2
    
    # Penalty weights
    PENALTY_UNSERVED = 10000
    PENALTY_TIME_VIOLATION = 5000
    PENALTY_CAPACITY_VIOLATION = 3000
    
    def __init__(self, data_file: str = "data/vrp_instance.json", seed: Optional[int] = None):
        """
        Initialize the GA Solver.
        
        Args:
            data_file: Path to the VRP instance JSON file
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.data = self._load_data(data_file)
        self.nodes = self.data['nodes']
        self.fleet = self.data['fleet']
        self.map_size = self.data['meta']['map_size']
        
        # Separate nodes by terrain type
        self.ground_nodes = [n for n in self.nodes if n['id'] != 0 and n['terrain'] == 0]
        self.uav_nodes = [n for n in self.nodes if n['id'] != 0 and n['terrain'] == 1]
        
        # Precompute matrices
        self._compute_distance_matrix()
        self._compute_time_matrices()
        
        # Best solution tracking
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        
        print(f"Solver initialized:")
        print(f"  - Ground nodes (MCS): {len(self.ground_nodes)}")
        print(f"  - UAV nodes: {len(self.uav_nodes)}")
        print(f"  - MCS fleet size: {self.fleet['MCS']['count']}")
        print(f"  - UAV fleet size: {self.fleet['UAV']['count']}")
    
    def _load_data(self, filename: str) -> Dict[str, Any]:
        """Load VRP instance data from JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _compute_distance_matrix(self) -> None:
        """Compute Euclidean distance matrix between all nodes."""
        n = len(self.nodes)
        self.distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.nodes[i]['x'] - self.nodes[j]['x']
                    dy = self.nodes[i]['y'] - self.nodes[j]['y']
                    self.distance_matrix[i][j] = math.sqrt(dx**2 + dy**2)
    
    def _compute_time_matrices(self) -> None:
        """Compute travel time matrices for MCS and UAV."""
        n = len(self.nodes)
        speed_mcs = self.fleet['MCS']['speed']
        speed_uav = self.fleet['UAV']['speed']
        
        self.time_matrix_mcs = self.distance_matrix / speed_mcs
        self.time_matrix_uav = self.distance_matrix / speed_uav
    
    def _get_node_by_id(self, node_id: int) -> Dict[str, Any]:
        """Get node data by ID."""
        return self.nodes[node_id]
    
    # ==================== Solution Representation ====================
    
    def _create_random_chromosome(self) -> Dict[str, List[int]]:
        """
        Create a random chromosome with two separate sequences.
        
        Returns:
            Dictionary with 'mcs_sequence' and 'uav_sequence'
        """
        mcs_seq = [n['id'] for n in self.ground_nodes]
        uav_seq = [n['id'] for n in self.uav_nodes]
        
        random.shuffle(mcs_seq)
        random.shuffle(uav_seq)
        
        return {
            'mcs_sequence': mcs_seq,
            'uav_sequence': uav_seq
        }
    
    def _create_time_based_chromosome(self) -> Dict[str, List[int]]:
        """
        Create a chromosome sorted by urgency (Earliest Deadline First).
        加入一點隨機性以避免每次都產生完全一樣的解。
        """
        # 1. 取得節點
        mcs_nodes = self.ground_nodes.copy()
        uav_nodes = self.uav_nodes.copy()
        
        # 2. 依照 l_time (最晚時間) 進行排序
        # 這裡的 x['l_time'] 就是時間窗的結束時間
        mcs_nodes.sort(key=lambda x: x['l_time'])
        uav_nodes.sort(key=lambda x: x['l_time'])
        
        # 3. 提取 ID
        mcs_seq = [n['id'] for n in mcs_nodes]
        uav_seq = [n['id'] for n in uav_nodes]
        
        # 4. (選用) 局部擾動：為了不讓所有聰明解都一模一樣，可以交換幾個鄰居
        # 這裡簡單示範：有 30% 機率對序列做輕微擾動
        if random.random() < 0.3:
            self._swap_mutation(mcs_seq)
            self._swap_mutation(uav_seq)

        return {
            'mcs_sequence': mcs_seq,
            'uav_sequence': uav_seq
        }
    
    def _initialize_population(self) -> List[Dict[str, List[int]]]:
        """
        Initialize population with a mix of random and heuristic chromosomes.
        """
        population = []
        
        # 策略：由 20% 的啟發式解 (Smart) + 80% 的隨機解 (Random) 組成
        n_heuristic = int(self.POPULATION_SIZE * 0.2)
        n_random = self.POPULATION_SIZE - n_heuristic
        
        print(f"Initializing population: {n_heuristic} heuristic + {n_random} random individuals.")
        
        # 1. 加入基於時間窗排序的解 (讓 GA 贏在起跑點)
        for _ in range(n_heuristic):
            population.append(self._create_time_based_chromosome())
            
        # 2. 加入隨機解 (維持多樣性)
        for _ in range(n_random):
            population.append(self._create_random_chromosome())
            
        return population
    
    # ==================== Decoder Logic (Split Routes) ====================
    
    def _split_routes(self, chromosome: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        Convert chromosome sequences into valid vehicle routes using greedy split.
        
        This is the core decoder that transforms genotype to phenotype.
        
        Args:
            chromosome: Dictionary with 'mcs_sequence' and 'uav_sequence'
            
        Returns:
            Dictionary containing routes for each vehicle type and unserved nodes
        """
        routes = {
            'MCS': [],
            'UAV': [],
            'unserved': []
        }
        
        # Split MCS sequence
        mcs_routes, mcs_unserved = self._split_single_fleet(
            sequence=chromosome['mcs_sequence'],
            vehicle_type='MCS',
            max_vehicles=self.fleet['MCS']['count'],
            capacity=self.fleet['MCS']['capacity'],
            time_matrix=self.time_matrix_mcs,
            speed=self.fleet['MCS']['speed']
        )
        routes['MCS'] = mcs_routes
        routes['unserved'].extend(mcs_unserved)
        
        # Split UAV sequence
        uav_routes, uav_unserved = self._split_single_fleet(
            sequence=chromosome['uav_sequence'],
            vehicle_type='UAV',
            max_vehicles=self.fleet['UAV']['count'],
            capacity=self.fleet['UAV']['capacity'],
            time_matrix=self.time_matrix_uav,
            speed=self.fleet['UAV']['speed']
        )
        routes['UAV'] = uav_routes
        routes['unserved'].extend(uav_unserved)
        
        return routes
    
    def _split_single_fleet(
        self,
        sequence: List[int],
        vehicle_type: str,
        max_vehicles: int,
        capacity: float,
        time_matrix: np.ndarray,
        speed: float
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Split a single sequence into multiple vehicle routes.
        
        Args:
            sequence: Node IDs to visit
            vehicle_type: 'MCS' or 'UAV'
            max_vehicles: Maximum number of vehicles available
            capacity: Vehicle capacity (kWh)
            time_matrix: Travel time matrix for this vehicle type
            speed: Vehicle speed (km/min)
            
        Returns:
            Tuple of (list of routes, list of unserved node IDs)
        """
        if not sequence:
            return [], []
        
        routes = []
        unserved = []
        current_route = [0]  # Start at depot
        current_load = 0.0
        current_time = self.data['meta'].get('work_start_time', 480)  # Default 08:00
        current_position = 0  # Depot
        vehicle_count = 0
        
        for node_id in sequence:
            node = self._get_node_by_id(node_id)
            demand = node['demand']
            
            # Calculate travel time to this node
            travel_time = time_matrix[current_position][node_id]
            arrival_time = current_time + travel_time
            
            # Wait if arriving early
            start_service = max(arrival_time, node['r_time'])
            
            # Check constraints
            capacity_ok = (current_load + demand) <= capacity
            time_ok = start_service <= node['l_time']
            
            # Check if vehicle can return to depot after this node
            return_time = time_matrix[node_id][0]
            can_return = True  # Simplified; can add battery constraints here
            
            if capacity_ok and time_ok and can_return:
                # Add node to current route
                current_route.append(node_id)
                current_load += demand
                current_time = start_service + node['service_time']
                current_position = node_id
            else:
                # Close current route and start new one
                if len(current_route) > 1:  # Only save if route has customers
                    current_route.append(0)  # Return to depot
                    routes.append(current_route)
                    vehicle_count += 1
                
                # Check if we can start a new vehicle
                if vehicle_count < max_vehicles:
                    # Start new route with this node
                    current_route = [0]
                    current_load = 0.0
                    current_time = self.data['meta'].get('work_start_time', 480)
                    current_position = 0
                    
                    # Try to add the node to new route
                    travel_time = time_matrix[0][node_id]
                    arrival_time = current_time + travel_time
                    start_service = max(arrival_time, node['r_time'])
                    
                    if start_service <= node['l_time'] and demand <= capacity:
                        current_route.append(node_id)
                        current_load = demand
                        current_time = start_service + node['service_time']
                        current_position = node_id
                    else:
                        # Node cannot be served even with fresh vehicle
                        unserved.append(node_id)
                else:
                    # No more vehicles available
                    unserved.append(node_id)
        
        # Close last route if it has customers
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
        
        return routes, unserved
    
    # ==================== Fitness Evaluation ====================
    
    def evaluate_solution(self, chromosome: Dict[str, List[int]]) -> float:
        """
        Evaluate the fitness of a chromosome (lower is better).
        
        Fitness = Total Travel Time + Total Waiting Time + Penalties
        
        Args:
            chromosome: The chromosome to evaluate
            
        Returns:
            Fitness value (cost)
        """
        routes = self._split_routes(chromosome)
        total_cost = 0.0
        
        # Evaluate MCS routes
        for route in routes['MCS']:
            cost = self._evaluate_route(route, 'MCS')
            total_cost += cost
        
        # Evaluate UAV routes
        for route in routes['UAV']:
            cost = self._evaluate_route(route, 'UAV')
            total_cost += cost
        
        # Add penalty for unserved customers
        total_cost += len(routes['unserved']) * self.PENALTY_UNSERVED
        
        return total_cost
    
    def _evaluate_route(self, route: List[int], vehicle_type: str) -> float:
        """
        Evaluate the cost of a single route.
        
        Args:
            route: List of node IDs including depot at start and end
            vehicle_type: 'MCS' or 'UAV'
            
        Returns:
            Route cost (travel time + waiting time)
        """
        if len(route) <= 2:  # Only depot
            return 0.0
        
        time_matrix = self.time_matrix_mcs if vehicle_type == 'MCS' else self.time_matrix_uav
        
        total_travel_time = 0.0
        total_waiting_time = 0.0
        current_time = self.data['meta'].get('work_start_time', 480)
        
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            
            # Travel time
            travel_time = time_matrix[from_node][to_node]
            total_travel_time += travel_time
            
            arrival_time = current_time + travel_time
            
            # Waiting time (if arriving before time window opens)
            if to_node != 0:  # Not depot
                node = self._get_node_by_id(to_node)
                if arrival_time < node['r_time']:
                    waiting_time = node['r_time'] - arrival_time
                    total_waiting_time += waiting_time
                    arrival_time = node['r_time']
                
                # Update current time (arrival + service)
                current_time = arrival_time + node['service_time']
            else:
                current_time = arrival_time
        
        return total_travel_time + total_waiting_time * 0.5  # Weight waiting time less
    
    # ==================== Genetic Operators ====================
    
    def _tournament_selection(self, population: List[Dict], fitnesses: List[float]) -> Dict:
        """
        Select a chromosome using tournament selection.
        
        Args:
            population: List of chromosomes
            fitnesses: Corresponding fitness values
            
        Returns:
            Selected chromosome (copy)
        """
        tournament_indices = random.sample(range(len(population)), self.TOURNAMENT_SIZE)
        best_idx = min(tournament_indices, key=lambda i: fitnesses[i])
        return copy.deepcopy(population[best_idx])
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Perform Order Crossover (OX) on two parent sequences.
        
        Args:
            parent1: First parent sequence
            parent2: Second parent sequence
            
        Returns:
            Tuple of two offspring sequences
        """
        if len(parent1) <= 2:
            return parent1.copy(), parent2.copy()
        
        size = len(parent1)
        
        # Select two crossover points
        point1, point2 = sorted(random.sample(range(size), 2))
        
        # Create offspring
        def create_offspring(p1: List[int], p2: List[int]) -> List[int]:
            offspring = [None] * size
            
            # Copy segment from p1
            offspring[point1:point2+1] = p1[point1:point2+1]
            
            # Fill remaining positions from p2
            p2_genes = [g for g in p2 if g not in offspring[point1:point2+1]]
            
            idx = 0
            for i in range(size):
                if offspring[i] is None:
                    offspring[i] = p2_genes[idx]
                    idx += 1
            
            return offspring
        
        child1 = create_offspring(parent1, parent2)
        child2 = create_offspring(parent2, parent1)
        
        return child1, child2
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        Apply crossover to two parent chromosomes.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of two offspring chromosomes
        """
        child1 = {'mcs_sequence': [], 'uav_sequence': []}
        child2 = {'mcs_sequence': [], 'uav_sequence': []}
        
        if random.random() < self.CROSSOVER_RATE:
            # Apply OX to MCS sequences
            if parent1['mcs_sequence'] and parent2['mcs_sequence']:
                child1['mcs_sequence'], child2['mcs_sequence'] = self._order_crossover(
                    parent1['mcs_sequence'], parent2['mcs_sequence']
                )
            else:
                child1['mcs_sequence'] = parent1['mcs_sequence'].copy()
                child2['mcs_sequence'] = parent2['mcs_sequence'].copy()
            
            # Apply OX to UAV sequences
            if parent1['uav_sequence'] and parent2['uav_sequence']:
                child1['uav_sequence'], child2['uav_sequence'] = self._order_crossover(
                    parent1['uav_sequence'], parent2['uav_sequence']
                )
            else:
                child1['uav_sequence'] = parent1['uav_sequence'].copy()
                child2['uav_sequence'] = parent2['uav_sequence'].copy()
        else:
            # No crossover, copy parents
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
        
        return child1, child2
    
    def _swap_mutation(self, sequence: List[int]) -> List[int]:
        """
        Apply swap mutation to a sequence.
        
        Args:
            sequence: Gene sequence
            
        Returns:
            Mutated sequence
        """
        if len(sequence) < 2:
            return sequence
        
        mutated = sequence.copy()
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def _mutate(self, chromosome: Dict) -> Dict:
        """
        Apply mutation to a chromosome.
        
        Args:
            chromosome: The chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        mutated = copy.deepcopy(chromosome)
        
        # Mutate MCS sequence
        if random.random() < self.MUTATION_RATE and mutated['mcs_sequence']:
            mutated['mcs_sequence'] = self._swap_mutation(mutated['mcs_sequence'])
        
        # Mutate UAV sequence
        if random.random() < self.MUTATION_RATE and mutated['uav_sequence']:
            mutated['uav_sequence'] = self._swap_mutation(mutated['uav_sequence'])
        
        return mutated
    
    # ==================== Main GA Loop ====================
    
    def solve(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the Genetic Algorithm to solve the VRP.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing best solution and statistics
        """
        print("\n" + "="*60)
        print("Starting Genetic Algorithm Optimization")
        print("="*60)
        
        # Initialize population
        population = self._initialize_population()
        
        # Evaluate initial population
        fitnesses = [self.evaluate_solution(chrom) for chrom in population]
        
        # Track best solution
        best_idx = np.argmin(fitnesses)
        self.best_solution = copy.deepcopy(population[best_idx])
        self.best_fitness = fitnesses[best_idx]
        self.fitness_history = [self.best_fitness]
        
        if verbose:
            print(f"Initial best fitness: {self.best_fitness:.2f}")
        
        # Main evolution loop
        for generation in range(self.MAX_GENERATIONS):
            new_population = []
            
            # Elitism: Keep best individuals
            sorted_indices = np.argsort(fitnesses)
            for i in range(self.ELITE_SIZE):
                new_population.append(copy.deepcopy(population[sorted_indices[i]]))
            
            # Generate offspring
            while len(new_population) < self.POPULATION_SIZE:
                # Selection
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.POPULATION_SIZE:
                    new_population.append(child2)
            
            # Replace population
            population = new_population
            
            # Evaluate new population
            fitnesses = [self.evaluate_solution(chrom) for chrom in population]
            
            # Update best solution
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < self.best_fitness:
                self.best_solution = copy.deepcopy(population[gen_best_idx])
                self.best_fitness = fitnesses[gen_best_idx]
            
            self.fitness_history.append(self.best_fitness)
            
            # Progress report
            if verbose and (generation + 1) % 50 == 0:
                avg_fitness = np.mean(fitnesses)
                print(f"Generation {generation + 1:4d}: Best = {self.best_fitness:.2f}, Avg = {avg_fitness:.2f}")
        
        print("\n" + "="*60)
        print("Optimization Complete!")
        print(f"Best fitness found: {self.best_fitness:.2f}")
        print("="*60)
        
        return self._get_solution_details()
    
    def _get_solution_details(self) -> Dict[str, Any]:
        """
        Get detailed information about the best solution.
        
        Returns:
            Dictionary with solution details
        """
        routes = self._split_routes(self.best_solution)
        
        # Calculate statistics
        total_distance = 0.0
        total_customers_served = 0
        
        for route in routes['MCS']:
            for i in range(len(route) - 1):
                total_distance += self.distance_matrix[route[i]][route[i+1]]
            total_customers_served += len(route) - 2  # Exclude depots
        
        for route in routes['UAV']:
            for i in range(len(route) - 1):
                total_distance += self.distance_matrix[route[i]][route[i+1]]
            total_customers_served += len(route) - 2
        
        return {
            'best_fitness': self.best_fitness,
            'chromosome': self.best_solution,
            'routes': routes,
            'total_distance': total_distance,
            'customers_served': total_customers_served,
            'unserved_count': len(routes['unserved']),
            'mcs_routes_count': len(routes['MCS']),
            'uav_routes_count': len(routes['UAV']),
            'fitness_history': self.fitness_history
        }
    
    # ==================== Visualization ====================
    
    def plot_convergence(self, output_file: str = "image/convergence.png") -> None:
        """
        Plot the convergence curve (fitness vs generations).
        
        Args:
            output_file: Output file path for the plot
        """
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, 'b-', linewidth=1.5, label='Best Fitness')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness (Cost)', fontsize=12)
        plt.title('GA Convergence Curve', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Convergence plot saved to {output_file}")
    
    def plot_routes(self, output_file: str = "image/solution_routes.png") -> None:
        """
        Visualize the solution routes.
        
        Args:
            output_file: Output file path for the plot
        """
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if self.best_solution is None:
            print("No solution available. Run solve() first.")
            return
        
        routes = self._split_routes(self.best_solution)
        
        plt.figure(figsize=(12, 10))
        
        # Color palettes
        mcs_colors = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(routes['MCS']), 1)))
        uav_colors = plt.cm.Reds(np.linspace(0.4, 0.9, max(len(routes['UAV']), 1)))
        
        # Plot depot
        depot = self.nodes[0]
        plt.scatter(depot['x'], depot['y'], c='black', marker='s', s=200, 
                   zorder=5, label='Depot', edgecolors='gold', linewidths=2)
        
        # Plot MCS routes
        for idx, route in enumerate(routes['MCS']):
            color = mcs_colors[idx]
            x_coords = [self.nodes[node_id]['x'] for node_id in route]
            y_coords = [self.nodes[node_id]['y'] for node_id in route]
            
            plt.plot(x_coords, y_coords, 'o-', color=color, linewidth=2, 
                    markersize=8, label=f'MCS Route {idx+1}')
            
            # Annotate nodes
            for node_id in route[1:-1]:
                node = self.nodes[node_id]
                plt.annotate(str(node_id), (node['x'], node['y']), 
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
        
        # Plot UAV routes
        for idx, route in enumerate(routes['UAV']):
            color = uav_colors[idx]
            x_coords = [self.nodes[node_id]['x'] for node_id in route]
            y_coords = [self.nodes[node_id]['y'] for node_id in route]
            
            plt.plot(x_coords, y_coords, '^--', color=color, linewidth=2, 
                    markersize=10, label=f'UAV Route {idx+1}')
            
            # Annotate nodes
            for node_id in route[1:-1]:
                node = self.nodes[node_id]
                plt.annotate(str(node_id), (node['x'], node['y']), 
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
        
        # Plot unserved nodes
        if routes['unserved']:
            unserved_x = [self.nodes[nid]['x'] for nid in routes['unserved']]
            unserved_y = [self.nodes[nid]['y'] for nid in routes['unserved']]
            plt.scatter(unserved_x, unserved_y, c='gray', marker='x', s=100, 
                       linewidths=2, label='Unserved', zorder=4)
        
        plt.xlabel('X (km)', fontsize=12)
        plt.ylabel('Y (km)', fontsize=12)
        plt.title(f'VRP Solution Routes (Cost: {self.best_fitness:.2f})', fontsize=14)
        plt.legend(loc='upper right', fontsize=9)
        plt.xlim(-5, self.map_size + 5)
        plt.ylim(-5, self.map_size + 5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Route visualization saved to {output_file}")
    
    def print_solution_summary(self) -> None:
        """Print a detailed summary of the best solution."""
        if self.best_solution is None:
            print("No solution available. Run solve() first.")
            return
        
        routes = self._split_routes(self.best_solution)
        
        print("\n" + "="*60)
        print("SOLUTION SUMMARY")
        print("="*60)
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Cost (Fitness): {self.best_fitness:.2f}")
        print(f"   MCS Routes Used: {len(routes['MCS'])} / {self.fleet['MCS']['count']}")
        print(f"   UAV Routes Used: {len(routes['UAV'])} / {self.fleet['UAV']['count']}")
        print(f"   Unserved Customers: {len(routes['unserved'])}")
        
        print(f"\n🚛 MCS Routes:")
        for idx, route in enumerate(routes['MCS']):
            route_demand = sum(self.nodes[nid]['demand'] for nid in route[1:-1])
            route_distance = sum(
                self.distance_matrix[route[i]][route[i+1]] 
                for i in range(len(route)-1)
            )
            print(f"   Route {idx+1}: {route}")
            print(f"           Demand: {route_demand:.1f} kWh, Distance: {route_distance:.2f} km")
        
        print(f"\n🚁 UAV Routes:")
        for idx, route in enumerate(routes['UAV']):
            route_demand = sum(self.nodes[nid]['demand'] for nid in route[1:-1])
            route_distance = sum(
                self.distance_matrix[route[i]][route[i+1]] 
                for i in range(len(route)-1)
            )
            print(f"   Route {idx+1}: {route}")
            print(f"           Demand: {route_demand:.1f} kWh, Distance: {route_distance:.2f} km")
        
        if routes['unserved']:
            print(f"\n⚠️  Unserved Nodes: {routes['unserved']}")
        
        print("\n" + "="*60)


# ==================== Main Execution ====================

if __name__ == "__main__":
    # Initialize solver with the generated instance
    solver = GASolver(data_file="data/vrp_instance.json", seed=42)
    
    # Run the genetic algorithm
    solution = solver.solve(verbose=True)
    
    # Print detailed solution summary
    solver.print_solution_summary()
    
    # Generate visualizations
    solver.plot_convergence("image/convergence.png")
    solver.plot_routes("image/solution_routes.png")
    
    print("\n✅ Optimization complete! Check the 'image/' folder for visualizations.")
