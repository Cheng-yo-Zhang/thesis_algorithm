class ALNSConfig:
    """Adaptive Large Neighborhood Search 超參數配置"""
    # --- 迭代控制 ---
    MAX_ITERATIONS: int = 5000
    SEGMENT_SIZE: int = 100           # 每 segment 更新一次權重

    # --- Simulated Annealing ---
    SA_INITIAL_TEMP: float = 100.0    # 初始溫度
    SA_COOLING_RATE: float = 0.9995   # 冷卻係數 (每 iteration)
    SA_FINAL_TEMP: float = 0.01      # 終止溫度

    # --- 算子獎勵分數 ---
    SIGMA_1: float = 33.0            # 找到新全局最佳解
    SIGMA_2: float = 13.0            # 比當前解好 (被接受)
    SIGMA_3: float = 5.0             # 比當前解差但被 SA 接受

    # --- Destroy ---
    DESTROY_RATIO_MIN: float = 0.1   # 最少摧毀比例
    DESTROY_RATIO_MAX: float = 0.4   # 最多摧毀比例
    WORST_REMOVAL_P: float = 3.0     # Worst Removal 隨機性參數 (越大越 deterministic)
    SHAW_REMOVAL_P: float = 6.0      # Shaw Removal 隨機性參數 (越大越 deterministic)

    # --- Shaw Relatedness 權重 (Ropke & Pisinger, 2006) ---
    SHAW_WEIGHT_DIST: float = 9.0    # 距離相似度權重 (φ₁)
    SHAW_WEIGHT_TIME: float = 3.0    # 時間窗相似度權重 (φ₂)
    SHAW_WEIGHT_DEMAND: float = 2.0  # 需求量相似度權重 (φ₃)

    # --- 算子權重 ---
    REACTION_FACTOR: float = 0.1     # 權重更新的反應係數 λ