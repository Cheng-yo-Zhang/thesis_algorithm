"""
Instance Generator for EV Mobile Charging Scheduling Problem
=============================================================

CLI-based generator for physically-consistent test instances.

Usage:
    python instance_generator.py

Features:
- Interactive CLI for parameter input
- Spatial distributions: Random (R), Clustered (C), Random-Clustered (RC)
- Urgent customers: 30-60 min time windows (fast charging)
- Normal customers: 8-10 hour time windows (overnight charging)
- Output compatible with existing main.py CSV format

Author: [Your Name]
Date: 2026-02
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import os


# ==================== Physical Constants ====================

@dataclass
class EVChargingPhysics:
    """Real-world EV charging parameters (all units in minutes and km)"""
    
    # Charging power (kW)
    FAST_CHARGE_POWER: float = 250.0    # DC fast charging
    SLOW_CHARGE_POWER: float = 11.0      # AC slow charging
    
    # Typical energy demands (kWh)
    DEMAND_LOW: Tuple[int, int] = (5, 15)      # High SOC, just top-up
    DEMAND_MEDIUM: Tuple[int, int] = (15, 30)  # Medium SOC
    DEMAND_HIGH: Tuple[int, int] = (30, 50)    # Low SOC, needs full charge
    
    # Service overhead (minutes)
    SETUP_TIME: float = 3.0          # Vehicle positioning, cable connection
    TEARDOWN_TIME: float = 2.0       # Disconnection, payment
    
    def fast_charge_time(self, demand_kwh: float) -> float:
        """Calculate fast charging time in minutes"""
        efficiency = 0.92
        charge_time = (demand_kwh / self.FAST_CHARGE_POWER) * 60 / efficiency
        return charge_time + self.SETUP_TIME + self.TEARDOWN_TIME
    
    def slow_charge_time(self, demand_kwh: float) -> float:
        """Calculate slow charging time in minutes"""
        efficiency = 0.95
        charge_time = (demand_kwh / self.SLOW_CHARGE_POWER) * 60 / efficiency
        return charge_time + self.SETUP_TIME + self.TEARDOWN_TIME


@dataclass
class InstanceConfig:
    """Configuration for instance generation"""
    
    # Instance identification
    name: str = "instance"
    
    # Size parameters
    n_customers: int = 50
    
    # Spatial parameters
    spatial_type: str = "random"  # "random", "clustered", "rc"
    area_size: int = 100          # 100 km × 100 km (fixed)
    n_clusters: int = 5           # For clustered/rc types
    cluster_std: float = 10.0     # Standard deviation within clusters (km)
    
    # Planning horizon (minutes) - 24 hours for overnight scenarios
    planning_horizon: int = 1440  # 24 hours = 1440 minutes
    
    # Customer mix
    urgent_ratio: float = 0.3     # 30% urgent customers
    
    # Time window settings (in minutes)
    # Urgent: 30-60 minutes (fast charging scenarios)
    tw_urgent_min: int = 60
    tw_urgent_max: int = 90
    
    # Normal: 8-10 hours overnight (480-600 minutes)
    tw_normal_min: int = 480
    tw_normal_max: int = 600
    
    # Demand distribution (probability weights)
    demand_distribution: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.30,      # 30% high SOC customers
        'medium': 0.50,   # 50% medium SOC
        'high': 0.20,     # 20% low SOC (need more charging)
    })
    
    # Depot location (center of 100km × 100km area)
    depot_x: int = 50
    depot_y: int = 50
    
    # Random seed for reproducibility
    seed: int = 42


# ==================== Instance Generator ====================

class InstanceGenerator:
    """Generate test instances for EV Mobile Charging Scheduling"""
    
    def __init__(self, config: InstanceConfig):
        self.config = config
        self.physics = EVChargingPhysics()
        np.random.seed(config.seed)
    
    def generate(self) -> pd.DataFrame:
        """Generate a complete instance"""
        
        # Step 1: Generate customer locations (integer coordinates)
        locations = self._generate_locations()
        
        # Step 2: Assign node types (urgent vs normal)
        node_types = self._assign_node_types()
        
        # Step 3: Generate demands (integer kWh)
        demands = self._generate_demands()
        
        # Step 4: Calculate service times based on demands and types
        service_times = self._calculate_service_times(demands, node_types)
        
        # Step 5: Generate time windows based on node types
        ready_times, due_dates = self._generate_time_windows(
            locations, node_types, service_times
        )
        
        # Step 6: Build DataFrame
        df = self._build_dataframe(
            locations, demands, ready_times, due_dates, service_times, node_types
        )
        
        return df
    
    def _generate_locations(self) -> np.ndarray:
        """Generate customer locations based on spatial type (integer coordinates)"""
        n = self.config.n_customers
        size = self.config.area_size
        
        if self.config.spatial_type == "random":
            locations = np.random.randint(0, size + 1, (n, 2))
            
        elif self.config.spatial_type == "clustered":
            locations = self._generate_clustered_locations(n, size, cluster_ratio=1.0)
            
        elif self.config.spatial_type == "rc":
            locations = self._generate_clustered_locations(n, size, cluster_ratio=0.5)
            
        else:
            raise ValueError(f"Unknown spatial type: {self.config.spatial_type}")
        
        # Clip to valid range and convert to int
        locations = np.clip(locations, 0, size).astype(int)
        return locations
    
    def _generate_clustered_locations(
        self, n: int, size: int, cluster_ratio: float
    ) -> np.ndarray:
        """Generate locations with some clustered and some random"""
        
        n_clustered = int(n * cluster_ratio)
        n_random = n - n_clustered
        
        locations = []
        
        # Generate cluster centers (avoid edges)
        n_clusters = self.config.n_clusters
        centers = np.random.randint(
            int(size * 0.15), int(size * 0.85), (n_clusters, 2)
        )
        
        # Assign customers to clusters
        cluster_assignments = np.random.randint(0, n_clusters, n_clustered)
        
        for i in range(n_clustered):
            center = centers[cluster_assignments[i]]
            offset = np.random.normal(0, self.config.cluster_std, 2)
            loc = center + offset
            locations.append(loc)
        
        # Add random customers
        if n_random > 0:
            random_locs = np.random.randint(0, size + 1, (n_random, 2))
            locations.extend(random_locs)
        
        np.random.shuffle(locations)
        return np.array(locations)
    
    def _assign_node_types(self) -> List[str]:
        """Assign urgent or normal type to each customer"""
        n = self.config.n_customers
        n_urgent = int(n * self.config.urgent_ratio)
        
        types = ['urgent'] * n_urgent + ['normal'] * (n - n_urgent)
        np.random.shuffle(types)
        return types
    
    def _generate_demands(self) -> np.ndarray:
        """Generate energy demands (integer kWh) based on SOC distribution"""
        n = self.config.n_customers
        demands = []
        
        dist = self.config.demand_distribution
        categories = np.random.choice(
            ['low', 'medium', 'high'],
            size=n,
            p=[dist['low'], dist['medium'], dist['high']]
        )
        
        for cat in categories:
            if cat == 'low':
                demand = np.random.randint(
                    self.physics.DEMAND_LOW[0], 
                    self.physics.DEMAND_LOW[1] + 1
                )
            elif cat == 'medium':
                demand = np.random.randint(
                    self.physics.DEMAND_MEDIUM[0], 
                    self.physics.DEMAND_MEDIUM[1] + 1
                )
            else:
                demand = np.random.randint(
                    self.physics.DEMAND_HIGH[0], 
                    self.physics.DEMAND_HIGH[1] + 1
                )
            demands.append(demand)
        
        return np.array(demands, dtype=int)
    
    def _calculate_service_times(
        self, demands: np.ndarray, node_types: List[str]
    ) -> np.ndarray:
        """Calculate service time based on demand and charging type"""
        service_times = []
        
        for demand, ntype in zip(demands, node_types):
            if ntype == 'urgent':
                st = self.physics.fast_charge_time(demand)
            else:
                st = self.physics.slow_charge_time(demand)
            service_times.append(int(round(st)))
        
        return np.array(service_times, dtype=int)
    
    def _generate_time_windows(
        self,
        locations: np.ndarray,
        node_types: List[str],
        service_times: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate time windows based on node type"""
        
        n = self.config.n_customers
        horizon = self.config.planning_horizon
        depot = np.array([self.config.depot_x, self.config.depot_y])
        
        ready_times = []
        due_dates = []
        
        for i in range(n):
            loc = locations[i]
            ntype = node_types[i]
            st = service_times[i]
            
            # Calculate minimum travel time from depot
            distance = np.linalg.norm(loc - depot)
            mcs_speed_km_per_min = 40.0 / 60.0  # 40 km/h
            min_travel_time = distance / mcs_speed_km_per_min
            
            # Set time window width based on node type
            if ntype == 'urgent':
                # Urgent: 30-60 minutes window
                tw_width = np.random.randint(
                    self.config.tw_urgent_min,
                    self.config.tw_urgent_max + 1
                )
            else:
                # Normal: 8-10 hours overnight window (480-600 minutes)
                tw_width = np.random.randint(
                    self.config.tw_normal_min,
                    self.config.tw_normal_max + 1
                )
            
            # READY_TIME: when customer becomes available
            earliest_ready = int(min_travel_time + 5)
            
            # For normal (overnight), ready time should be in the evening
            if ntype == 'normal':
                # Evening hours: 18:00-22:00 (1080-1320 minutes from midnight)
                # But we use relative time, so just spread across the horizon
                latest_ready = horizon - tw_width - st - int(min_travel_time)
                latest_ready = max(earliest_ready, latest_ready)
            else:
                # Urgent can appear anytime during the day
                latest_ready = horizon - tw_width - st - int(min_travel_time)
                latest_ready = max(earliest_ready, latest_ready)
            
            ready_time = np.random.randint(earliest_ready, max(earliest_ready + 1, latest_ready + 1))
            
            # DUE_DATE: deadline for service start + time window
            due_date = ready_time + tw_width
            
            # Ensure feasibility
            due_date = min(due_date, horizon)
            
            ready_times.append(ready_time)
            due_dates.append(due_date)
        
        return np.array(ready_times, dtype=int), np.array(due_dates, dtype=int)
    
    def _build_dataframe(
        self,
        locations: np.ndarray,
        demands: np.ndarray,
        ready_times: np.ndarray,
        due_dates: np.ndarray,
        service_times: np.ndarray,
        node_types: List[str]
    ) -> pd.DataFrame:
        """Build the output DataFrame with integer values"""
        
        # Create depot row
        depot_row = {
            'CUST NO.': 1,
            'XCOORD.': self.config.depot_x,
            'YCOORD.': self.config.depot_y,
            'DEMAND': 0,
            'READY TIME': 0,
            'DUE DATE': self.config.planning_horizon,
            'SERVICE TIME': 0,
            'NODE_TYPE': 'depot',
            'ORIGINAL_CUST_NO': 1
        }
        
        # Create customer rows
        customer_rows = []
        for i in range(self.config.n_customers):
            row = {
                'CUST NO.': i + 2,
                'XCOORD.': int(locations[i, 0]),
                'YCOORD.': int(locations[i, 1]),
                'DEMAND': int(demands[i]),
                'READY TIME': int(ready_times[i]),
                'DUE DATE': int(due_dates[i]),
                'SERVICE TIME': int(service_times[i]),
                'NODE_TYPE': node_types[i],
                'ORIGINAL_CUST_NO': i + 2
            }
            customer_rows.append(row)
        
        all_rows = [depot_row] + customer_rows
        df = pd.DataFrame(all_rows)
        
        return df
    
    def save(self, df: pd.DataFrame, filepath: str):
        """Save instance to CSV file"""
        df.to_csv(filepath, index=False)
        print(f"\n✅ Instance saved to: {filepath}")
    
    def print_summary(self, df: pd.DataFrame):
        """Print instance summary statistics"""
        customers = df[df['NODE_TYPE'] != 'depot']
        
        print("\n" + "="*60)
        print(f"📊 Instance Summary: {self.config.name}")
        print("="*60)
        print(f"Total customers:    {len(customers)}")
        print(f"Spatial type:       {self.config.spatial_type}")
        print(f"Area size:          {self.config.area_size} km × {self.config.area_size} km")
        print(f"Planning horizon:   {self.config.planning_horizon} minutes ({self.config.planning_horizon/60:.0f} hours)")
        print()
        
        # Node type distribution
        urgent_count = len(customers[customers['NODE_TYPE'] == 'urgent'])
        normal_count = len(customers[customers['NODE_TYPE'] == 'normal'])
        print(f"🔴 Urgent customers: {urgent_count} ({urgent_count/len(customers)*100:.1f}%)")
        print(f"🟢 Normal customers: {normal_count} ({normal_count/len(customers)*100:.1f}%)")
        print()
        
        # Time window statistics by type
        urgent_tw = customers[customers['NODE_TYPE'] == 'urgent']
        normal_tw = customers[customers['NODE_TYPE'] == 'normal']
        
        if len(urgent_tw) > 0:
            urgent_widths = urgent_tw['DUE DATE'] - urgent_tw['READY TIME']
            print(f"⏱️  Urgent time windows: {urgent_widths.min():.0f}-{urgent_widths.max():.0f} min (mean: {urgent_widths.mean():.0f})")
        
        if len(normal_tw) > 0:
            normal_widths = normal_tw['DUE DATE'] - normal_tw['READY TIME']
            print(f"⏱️  Normal time windows: {normal_widths.min():.0f}-{normal_widths.max():.0f} min ({normal_widths.mean()/60:.1f} hours avg)")
        print()
        
        # Service time statistics
        print(f"Service time range: {customers['SERVICE TIME'].min():.0f}-{customers['SERVICE TIME'].max():.0f} min")
        print(f"Demand range:       {customers['DEMAND'].min():.0f}-{customers['DEMAND'].max():.0f} kWh")
        print("="*60)


# ==================== CLI Interface ====================

def get_user_input():
    """Interactive CLI to get instance parameters from user"""
    
    print("\n" + "="*60)
    print("🔋 EV Mobile Charging Instance Generator")
    print("="*60)
    print("Generate test instances for scheduling optimization")
    print("-"*60)
    
    # Number of customers
    while True:
        try:
            n_customers = int(input("\n📍 Number of customers [default=50]: ").strip() or "50")
            if n_customers < 1:
                print("   ❌ Must be at least 1 customer")
                continue
            break
        except ValueError:
            print("   ❌ Please enter a valid integer")
    
    # Urgent ratio
    while True:
        try:
            urgent_pct = float(input("🔴 Urgent customer ratio (0-100%) [default=30]: ").strip() or "30")
            if not 0 <= urgent_pct <= 100:
                print("   ❌ Must be between 0 and 100")
                continue
            urgent_ratio = urgent_pct / 100.0
            break
        except ValueError:
            print("   ❌ Please enter a valid number")
    
    normal_ratio = 1.0 - urgent_ratio
    print(f"🟢 Normal customer ratio: {normal_ratio*100:.1f}%")
    
    # Spatial distribution
    print("\n📊 Spatial distribution types:")
    print("   [1] Random    - Uniform distribution across area")
    print("   [2] Clustered - Customers grouped in clusters")
    print("   [3] RC        - Mixed (50% clustered, 50% random)")
    
    while True:
        spatial_choice = input("   Select type [1/2/3, default=1]: ").strip() or "1"
        spatial_map = {"1": "random", "2": "clustered", "3": "rc"}
        if spatial_choice in spatial_map:
            spatial_type = spatial_map[spatial_choice]
            break
        print("   ❌ Please enter 1, 2, or 3")
    
    # Random seed
    while True:
        try:
            seed = int(input("\n🎲 Random seed [default=42]: ").strip() or "42")
            break
        except ValueError:
            print("   ❌ Please enter a valid integer")
    
    # Output filename
    default_name = f"instance_{n_customers}c_{spatial_type}_s{seed}"
    output_name = input(f"\n📁 Output filename [default={default_name}]: ").strip() or default_name
    if not output_name.endswith('.csv'):
        output_name += '.csv'
    
    return {
        'n_customers': n_customers,
        'urgent_ratio': urgent_ratio,
        'spatial_type': spatial_type,
        'seed': seed,
        'output_name': output_name
    }


def main():
    """Main CLI entry point"""
    
    # Get user input
    params = get_user_input()
    
    # Create configuration
    config = InstanceConfig(
        name=params['output_name'].replace('.csv', ''),
        n_customers=params['n_customers'],
        spatial_type=params['spatial_type'],
        urgent_ratio=params['urgent_ratio'],
        seed=params['seed'],
        # Fixed parameters
        area_size=100,           # 100km × 100km
        planning_horizon=1440,   # 24 hours
    )
    
    print("\n⏳ Generating instance...")
    
    # Generate instance
    generator = InstanceGenerator(config)
    df = generator.generate()
    
    # Print summary
    generator.print_summary(df)
    
    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), params['output_name'])
    generator.save(df, output_path)
    
    # Preview
    print("\n📋 Preview (first 10 rows):")
    print("-"*60)
    print(df.head(10).to_string(index=False))
    print("-"*60)
    
    print("\n✨ Done! You can now use this instance with main.py")


if __name__ == "__main__":
    main()
