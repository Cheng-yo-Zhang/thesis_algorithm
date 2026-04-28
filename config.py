from dataclasses import dataclass, field


@dataclass
class Config:
    """統一實驗配置 (static single-batch) — 所有實驗變數與系統參數集中管理"""

    # === 實驗變數 ===
    RANDOM_SEED: int = 42
    N_REQUESTS: int = 10                       # 一次性批次中產生的請求數
    URGENT_RATIO: float = 0.5                  # Urgent 需求佔比 ρ
    CONSTRUCTION_STRATEGY: str = "regret2"     # "edf" | "slack" | "regret2" | "nearest"
    DELTA_T: int = 15                          # 請求 ready_time 的散佈寬度 (min)

    # === 服務區域 ===
    AREA_SIZE: float = 20.0                    # 20 km × 20 km
    DEPOT_X: float = 10.0
    DEPOT_Y: float = 10.0

    # === 需求分布 ===
    DEMAND_MEAN: float = 6.9                   # kWh
    DEMAND_STD: float = 4.9                    # kWh
    URGENT_ENERGY_MIN: float = 2.0             # kWh
    URGENT_ENERGY_MAX: float = 20.0
    NORMAL_ENERGY_MIN: float = 2.0
    NORMAL_ENERGY_MAX: float = 20.0
    URGENT_TW_MIN: float = 30.0                # min
    URGENT_TW_MAX: float = 60.0
    NORMAL_TW_MIN: float = 480.0               # min
    NORMAL_TW_MAX: float = 600.0

    # === 車隊配置 ===
    NUM_MCS_SLOW: int = 20
    NUM_MCS_FAST: int = 20
    NUM_UAV: int = 20

    # === 能量模型 ===
    MCS_UNLIMITED_ENERGY: bool = True          # True = MCS 無限能量

    # === MCS 共用參數 ===
    MCS_SPEED: float = 0.72                    # km/min (43.2 km/h)
    MCS_CAPACITY: float = 100.0                # kWh
    MCS_ENERGY_CONSUMPTION: float = 0.5        # kWh/km
    MCS_ARRIVAL_SETUP_MIN: float = 5.0
    MCS_CONNECT_MIN: float = 2.0
    MCS_DISCONNECT_MIN: float = 2.0

    # === MCS 充電功率 ===
    MCS_SLOW_POWER: float = 22.0               # kW
    MCS_FAST_POWER: float = 100.0              # kW

    # === UAV 參數 ===
    UAV_SPEED: float = 1.5                     # km/min (60 km/h)
    UAV_ENDURANCE: float = 360.0               # min
    UAV_MAX_RANGE: float = 200.0               # km
    UAV_CHARGE_POWER: float = 50.0             # kW
    UAV_DELIVERABLE_ENERGY: float = 20.0       # kWh
    UAV_TAKEOFF_LAND_OVERHEAD: float = 3.0     # min

    # === UAV Safe-SOC ===
    UAV_TARGET_SOC: float = 0.30
    EV_BATTERY_CAPACITY: float = 60.0          # kWh

    # === EV 初始 SOC 取樣 (urgent only) ===
    URGENT_SOC_MIN: float = 0.05
    URGENT_SOC_MAX: float = 0.20

    # === Urgent 角落分布 ===
    URGENT_CORNER_RATIO: float = 0.3
    URGENT_CORNER_SIGMA: float = 2.0           # km
    CORNER_POSITIONS: list = field(default_factory=lambda: [
        (1.0, 1.0), (1.0, 19.0), (19.0, 1.0), (19.0, 19.0)
    ])

    # === Type-Flexible Construction 偏好獎勵 ===
    SLOW_PREFERENCE_BONUS: float = 5.0
    UAV_URGENT_BONUS: float = 10.0

    # === 目標函數權重 ===
    # Z = α_u·N_miss_u + α_n·N_miss_n + β_u·W_urgent + β_n·W_normal + γ·D_total
    ALPHA_URGENT: float = 3500.0
    ALPHA_NORMAL: float = 1000.0
    BETA_WAITING_URGENT: float = 5.0
    BETA_WAITING_NORMAL: float = 0.5
    GAMMA_DISTANCE: float = 1.0

    # === ALNS 參數 ===
    ALNS_MAX_ITERATIONS: int = 5000
    SA_INITIAL_TEMP: float = 200.0
    SA_COOLING_RATE: float = 0.9995
    ALNS_REMOVAL_MIN: int = 3
    ALNS_REMOVAL_MAX: int = 10
    ALNS_MAX_COVERAGE_LOSS: int = 2
    ALNS_SEGMENT_SIZE: int = 100
    ALNS_REACTION_FACTOR: float = 0.1
    ALNS_SIGMA1: float = 33.0
    ALNS_SIGMA2: float = 9.0
    ALNS_SIGMA3: float = 3.0
    LAMBDA_BALANCE: float = 5.0
    ENABLE_CROSS_FLEET_LS: bool = False
