from dataclasses import dataclass, field


@dataclass
class Config:
    """統一實驗配置 — 所有實驗變數與系統參數集中管理"""

    # === 實驗變數 (Experimental Variables) ===
    RANDOM_SEED: int = 42
    USE_HPP: bool = True              # True = HPP (固定到達率), False = NHPP (時變到達率)
    USE_FIXED_DEMAND: bool = True      # True = 每 slot 固定需求數 (不使用 Poisson)
    FIXED_DEMAND_PER_SLOT: int = 10    # 固定需求數量
    LAMBDA_BAR: float = 0.15             # 平均到達率 (requests/min), 約 9 req/hr
    URGENT_RATIO: float = 0.5          # Urgent 需求佔比 ρ
    CONSTRUCTION_STRATEGY: str = "regret2"  # "edf" | "slack" | "regret2" | "nearest"
    DELTA_T: int = 15                  # 排程週期 (min)
    T_TOTAL: int = 2880               # 模擬總時長 (min) — 48 小時

    # === 測量窗口 (Measurement Window) ===
    WARMUP_END: int = 720             # warm-up 結束 (t=720, 第 12 小時)
    MEASUREMENT_START: int = 720      # 測量開始
    MEASUREMENT_END: int = 2160       # 測量結束 (t=2160, 第 36 小時)
    GENERATION_END: int = 0           # 停止生成請求的時間 (0 = 不限制，跑到 T_TOTAL)

    # === 實驗參數 ===
    TARGET_SERVICE_RATE: float = 0.95
    NUM_REPLICATIONS: int = 10

    # === 視覺化 ===
    PLOT_MAX_SLOTS: int = 10           # 最多畫幾個 slot 的圖 (0 = 不畫)

    # === NHPP 24hr 輪廓 r(t) — [(boundary_min, rate), ...] ===
    # 深夜 0.3 | 早高峰 1.2 | 上午 0.8 | 午間 1.4 | 下午 1.0 | 晚高峰 1.5 | 晚間 0.5
    NHPP_PROFILE: list = field(default_factory=lambda: [
        (360,  0.3),   # 0:00-6:00  深夜
        (540,  1.2),   # 6:00-9:00  早高峰
        (720,  0.8),   # 9:00-12:00 上午
        (840,  1.4),   # 12:00-14:00 午間
        (1020, 1.0),   # 14:00-17:00 下午
        (1200, 1.5),   # 17:00-20:00 晚高峰
        (1440, 0.5),   # 20:00-24:00 晚間
    ])

    # === 服務區域 ===
    AREA_SIZE: float = 20.0            # 20 km × 20 km
    DEPOT_X: float = 10.0
    DEPOT_Y: float = 10.0

    # === 需求分布 ===
    DEMAND_MEAN: float = 6.9          # kWh
    DEMAND_STD: float = 4.9           # kWh
    URGENT_ENERGY_MIN: float = 2.0     # kWh
    URGENT_ENERGY_MAX: float = 20.0
    NORMAL_ENERGY_MIN: float = 2.0
    NORMAL_ENERGY_MAX: float = 20.0
    URGENT_TW_MIN: float = 30.0        # min (時間窗寬度)
    URGENT_TW_MAX: float = 60.0
    NORMAL_TW_MIN: float = 480.0       # min
    NORMAL_TW_MAX: float = 600.0

    # === 車隊配置 (初始 active fleet) — 給足夠大的 pool，讓 dispatch 自己決定用多少 ===
    NUM_MCS_SLOW: int = 20
    NUM_MCS_FAST: int = 20
    NUM_UAV: int = 20

    # === 能量模型 ===
    MCS_UNLIMITED_ENERGY: bool = True  # True = MCS 無限能量 (fleet sizing 實驗用)

    # === MCS 共用參數 ===
    MCS_SPEED: float = 0.72            # km/min (43.2 km/h)
    MCS_CAPACITY: float = 100.0        # kWh
    MCS_ENERGY_CONSUMPTION: float = 0.5  # kWh/km
    MCS_ARRIVAL_SETUP_MIN: float = 5.0   # 到達後 setup 時間 (min)
    MCS_CONNECT_MIN: float = 2.0         # 接線時間 (min)
    MCS_DISCONNECT_MIN: float = 2.0      # 拆線時間 (min)

    # === MCS 充電功率 ===
    MCS_SLOW_POWER: float = 22.0       # kW (AC Slow Charging)
    MCS_FAST_POWER: float = 100.0      # kW (DC Super Fast Charging)

    # === UAV 參數 ===
    UAV_SPEED: float = 1.5             # km/min (60 km/h)
    UAV_ENDURANCE: float = 360.0       # min (6 hr)
    UAV_MAX_RANGE: float = 200.0       # km
    UAV_CHARGE_POWER: float = 50.0     # kW
    UAV_DELIVERABLE_ENERGY: float = 20.0  # kWh
    UAV_TAKEOFF_LAND_OVERHEAD: float = 3.0  # min 起降作業開銷

    # === UAV 可服務節點篩選 ===
    UAV_ELIGIBLE_QUANTILE: float = 0.8  # (已棄用，改為全部可服務)

    # === UAV Safe-SOC 充電 ===
    UAV_TARGET_SOC: float = 0.30          # UAV 充到的目標 SOC (30%)
    EV_BATTERY_CAPACITY: float = 60.0    # EV 電池容量 (kWh)，用於 SOC 換算
    UAV_REQUEUE_FOR_MCS: bool = True     # UAV 服務後，剩餘需求重新排入 MCS

    # === EV 初始 SOC 取樣 ===
    URGENT_SOC_MIN: float = 0.05          # Urgent EV 初始 SOC 下限
    URGENT_SOC_MAX: float = 0.20          # Urgent EV 初始 SOC 上限

    # === Urgent 角落分布 ===
    URGENT_CORNER_RATIO: float = 0.3       # urgent 請求出現在角落的比例
    URGENT_CORNER_SIGMA: float = 2.0       # 角落高斯分布的 σ (km)
    CORNER_POSITIONS: list = field(default_factory=lambda: [
        (1.0, 1.0), (1.0, 19.0), (19.0, 1.0), (19.0, 19.0)
    ])

    # === Type-Flexible Construction 偏好獎勵 ===
    SLOW_PREFERENCE_BONUS: float = 5.0    # min — MCS-SLOW 經濟偏好獎勵
    UAV_URGENT_BONUS: float = 10.0        # min — UAV 服務 urgent 且 slack < 15min 的獎勵

    # === 目標函數權重 ===
    # Z = α_u·N_miss_u + α_n·N_miss_n + β_u·W_urgent + β_n·W_normal + γ·D_total
    #
    # 懲罰值推導 (基於問題規模):
    #   max_detour = 2 × AREA_SIZE × √2 ≈ 56.6 km
    #   max_single_cost = γ × max_detour + β_u × TW_urgent_max
    #                   = 1.0 × 56.6 + 5.0 × 60 ≈ 357
    #   α_u = 10 × max_single_cost ≈ 3500
    #   α_n =  3 × max_single_cost ≈ 1000
    ALPHA_URGENT: float = 3500.0       # Urgent 未服務懲罰 (10× 最大單趟成本)
    ALPHA_NORMAL: float = 1000.0       # Normal 未服務懲罰 ( 3× 最大單趟成本)
    BETA_WAITING_URGENT: float = 5.0   # Urgent 用戶回應時間權重 (min)
    BETA_WAITING_NORMAL: float = 0.5   # Normal 用戶回應時間權重 (min)
    GAMMA_DISTANCE: float = 1.0        # 行駛距離權重 (km)，對應油耗/電耗

    # === ALNS 參數 ===
    ALNS_MAX_ITERATIONS: int = 5000
    SA_INITIAL_TEMP: float = 200.0     # 校準至新 cost scale (典型 Δcost≈50-200)
    SA_COOLING_RATE: float = 0.9995    # Final T ≈ 200×0.082 = 16.4
    ALNS_REMOVAL_MIN: int = 3           # 每次最少移除節點數
    ALNS_REMOVAL_MAX: int = 10          # 每次最多移除節點數
    ALNS_MAX_COVERAGE_LOSS: int = 2     # 允許的最大 unassigned 增量
    ALNS_SEGMENT_SIZE: int = 100        # 權重更新週期
    ALNS_REACTION_FACTOR: float = 0.1   # 權重平滑因子 ρ
    ALNS_SIGMA1: float = 33.0           # 獎勵: new global best
    ALNS_SIGMA2: float = 9.0            # 獎勵: better than current
    ALNS_SIGMA3: float = 3.0            # 獎勵: SA accepted worse
    LAMBDA_BALANCE: float = 5.0          # ALNS load-aware penalty 權重 (內部用)
    ENABLE_CROSS_FLEET_LS: bool = False  # Cross-fleet local search

    def get_arrival_rate_profile(self, t: float) -> float:
        """取得時刻 t 的到達率輪廓 r(t)；HPP 模式下恆為 1.0"""
        if self.USE_HPP:
            return 1.0
        for boundary, rate in self.NHPP_PROFILE:
            if t < boundary:
                return rate
        return self.NHPP_PROFILE[-1][1]
