"""
NYC Yellow Taxi Trip Data → EV Charging Request Instance Generator (Grid-based)
================================================================================

This script reads the NYC Yellow Taxi trip dataset (Parquet format) and
the Taxi Zone shapefile, then:
  1. Projects zone centroids to UTM (meters).
  2. Maps each trip's O-D to grid coordinates (500 m cells, 100×100 grid).
  3. Applies physics-based filters (duration, distance, speed).
  4. Simulates EV battery depletion to generate charging requests.

Author : Louis
Date   : 2026-03
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


# =====================================================================
# 1. Configuration
# =====================================================================
@dataclass
class GridConfig:
    """所有可調參數集中管理。"""

    # --- I/O ---
    parquet_path: str = "yellow_tripdata_2025-01.parquet"
    shapefile_path: str = "taxi_zones.shp"

    # --- 時間切片 ---
    time_start: str = "2025-01-01 00:00:00"
    time_end: str = "2025-01-31 23:59:59"

    # --- 空間網格 ---
    center_lon: float = -73.9851        # 時代廣場經度
    center_lat: float = 40.7589         # 時代廣場緯度
    grid_half_size_m: float = 25_000.0  # 中心點到邊界距離 (m)
    cell_size_m: float = 500.0          # 每格邊長 (m)
    grid_cells: int = 100               # 網格每邊格數

    # --- 物理特徵過濾 ---
    min_duration_sec: float = 60.0
    max_duration_sec: float = 10_800.0
    min_speed_m_s: float = 0.28
    max_speed_m_s: float = 27.7

    # --- 電量模型 (Forced Trigger 模擬設定) ---
    battery_mean_kwh: float = 24.0      # 修改點：大幅降低平均電量，模擬已經跑一整天的車
    battery_std_kwh: float = 4.0        # 修改點：縮小標準差，集中在快沒電的區間
    battery_min_kwh: float = 18.1       # 初始電量下限 (確保出發時還沒觸發警報)
    battery_max_kwh: float = 90.0       # 滿電上限
    consumption_per_km: float = 0.5     # kWh/km
    charge_threshold_kwh: float = 18.0  # 觸發充電請求閾值
    random_seed: int = 42


# =====================================================================
# 2. NYCTaxiGridProcessor — 資料載入 / 網格化 / 物理過濾
# =====================================================================
class NYCTaxiGridProcessor:
    """將 TLC 旅程資料映射到公尺級網格並進行物理特徵過濾。"""

    def __init__(self, config: GridConfig) -> None:
        self.cfg = config

    # -----------------------------------------------------------------
    def _load_zone_centroids(self) -> pd.DataFrame:
        """載入 Taxi Zone Shapefile，投影至 UTM Zone 18N，回傳 zone → 公尺座標對應表。"""
        print("1. 載入 Taxi Zone Shapefile 並計算公尺級中心點...")
        gdf = gpd.read_file(self.cfg.shapefile_path)
        gdf = gdf.to_crs("EPSG:32618")
        gdf["centroid"] = gdf.geometry.centroid
        gdf["x_meters"] = gdf["centroid"].x
        gdf["y_meters"] = gdf["centroid"].y
        return gdf[["LocationID", "x_meters", "y_meters"]].set_index("LocationID")

    # -----------------------------------------------------------------
    def _load_trips(self) -> pd.DataFrame:
        """載入 Parquet 旅程資料並依時間範圍切片。"""
        print("2. 載入 TLC Parquet 資料與時間切片...")
        df = pd.read_parquet(self.cfg.parquet_path)
        df["pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
        mask = (
            (df["pickup_datetime"] >= self.cfg.time_start)
            & (df["pickup_datetime"] <= self.cfg.time_end)
        )
        return df[mask].copy()

    # -----------------------------------------------------------------
    def _map_coordinates(
        self, df: pd.DataFrame, zone_mapping: pd.DataFrame
    ) -> pd.DataFrame:
        """將 PU/DO LocationID 替換為 zone centroid 座標。"""
        print("3. 進行 O-D 座標映射 (Join)...")
        df = df.join(zone_mapping.add_prefix("PU_"), on="PULocationID")
        df = df.join(zone_mapping.add_prefix("DO_"), on="DOLocationID")
        return df.dropna(subset=["PU_x_meters", "DO_x_meters"]).copy()

    # -----------------------------------------------------------------
    def _apply_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """將公尺座標轉換為 (i, j) 網格索引，並過濾超出範圍的資料。"""
        print("4. 空間邊界框定與 100x100 網格化...")
        transformer = Transformer.from_crs("epsg:4326", "epsg:32618", always_xy=True)
        center_x, center_y = transformer.transform(
            self.cfg.center_lon, self.cfg.center_lat
        )
        min_x = center_x - self.cfg.grid_half_size_m
        min_y = center_y - self.cfg.grid_half_size_m
        cell = self.cfg.cell_size_m
        n = self.cfg.grid_cells

        df["pickup_i"] = np.floor((df["PU_x_meters"] - min_x) / cell).astype(int)
        df["pickup_j"] = np.floor((df["PU_y_meters"] - min_y) / cell).astype(int)
        df["dropoff_i"] = np.floor((df["DO_x_meters"] - min_x) / cell).astype(int)
        df["dropoff_j"] = np.floor((df["DO_y_meters"] - min_y) / cell).astype(int)

        mask = (
            df["pickup_i"].between(0, n - 1)
            & df["pickup_j"].between(0, n - 1)
            & df["dropoff_i"].between(0, n - 1)
            & df["dropoff_j"].between(0, n - 1)
        )
        return df[mask].copy()

    # -----------------------------------------------------------------
    def _filter_physics(self, df: pd.DataFrame) -> pd.DataFrame:
        """物理特徵過濾：時長、曼哈頓距離、速度。"""
        print("5. 物理特徵過濾 (防禦性過濾)...")
        cfg = self.cfg
        df["duration_sec"] = (
            (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()
        )
        df["manhattan_dist_m"] = (
            np.abs(df["pickup_i"] - df["dropoff_i"])
            + np.abs(df["pickup_j"] - df["dropoff_j"])
        ) * cfg.cell_size_m

        mask = (
            (df["duration_sec"] >= cfg.min_duration_sec)
            & (df["duration_sec"] <= cfg.max_duration_sec)
            & (df["manhattan_dist_m"] > 0)
        )
        df = df[mask].copy()

        df["speed_m_s"] = df["manhattan_dist_m"] / df["duration_sec"]
        df = df[
            (df["speed_m_s"] >= cfg.min_speed_m_s)
            & (df["speed_m_s"] <= cfg.max_speed_m_s)
        ].copy()

        print(f"清洗完成！剩餘有效軌跡數量: {len(df)}")
        return df

    # -----------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """
        執行完整 pipeline：
        Shapefile → Parquet → 座標映射 → 網格化 → 物理過濾。
        """
        zone_mapping = self._load_zone_centroids()
        df = self._load_trips()
        df = self._map_coordinates(df, zone_mapping)
        df = self._apply_grid(df)
        df = self._filter_physics(df)

        columns_to_keep = [
            "pickup_datetime", "dropoff_datetime", "duration_sec",
            "pickup_i", "pickup_j", "dropoff_i", "dropoff_j",
            "manhattan_dist_m",
        ]
        return df[columns_to_keep]


# =====================================================================
# 3. ChargingRequestGenerator — 充電請求模擬
# =====================================================================
class ChargingRequestGenerator:
    """基於背景流量模擬 EV 電量耗盡，產生充電請求 Task List。"""

    def __init__(self, config: GridConfig) -> None:
        self.cfg = config

    # -----------------------------------------------------------------
    def _assign_initial_battery(self, df: pd.DataFrame) -> pd.DataFrame:
        """依照 N(μ, σ²) 分配初始電量，截斷於 [min, max]。"""
        cfg = self.cfg
        print(f"1. 載入單日背景流量，共 {len(df)} 趟旅程...")
        np.random.seed(cfg.random_seed)
        print(f"2. 依照 N({cfg.battery_mean_kwh}, {cfg.battery_std_kwh}²) 分配初始電量...")
        df["initial_kwh"] = np.random.normal(
            loc=cfg.battery_mean_kwh, scale=cfg.battery_std_kwh, size=len(df)
        )
        df["initial_kwh"] = np.clip(
            df["initial_kwh"], a_min=cfg.battery_min_kwh, a_max=cfg.battery_max_kwh
        )
        return df

    # -----------------------------------------------------------------
    def _filter_triggered_evs(self, df: pd.DataFrame) -> pd.DataFrame:
        """篩選需電量 > 可用電量的 EV（半路觸發充電請求）。"""
        print("3. 計算耗電與篩選觸發 18 kWh 閾值的 EV...")
        cfg = self.cfg
        df["trip_dist_km"] = df["manhattan_dist_m"] / 1000.0
        df["needed_kwh"] = df["trip_dist_km"] * cfg.consumption_per_km
        df["usable_kwh"] = df["initial_kwh"] - cfg.charge_threshold_kwh

        mask = df["needed_kwh"] > df["usable_kwh"]
        df_out = df[mask].copy()
        print(f"   => 經過耗電模型篩選，共有 {len(df_out)} 台 EV 將發出充電請求！")
        return df_out

    # -----------------------------------------------------------------
    def _compute_request_location(self, df: pd.DataFrame) -> pd.DataFrame:
        """內插充電請求發生的確切時間與網格座標。"""
        print("4. 精確計算充電請求發生的時間與網格座標...")
        df["progress_ratio"] = df["usable_kwh"] / df["needed_kwh"]
        df["request_time"] = df["pickup_datetime"] + pd.to_timedelta(
            df["duration_sec"] * df["progress_ratio"], unit="s"
        )
        df["req_i"] = np.round(
            df["pickup_i"]
            + (df["dropoff_i"] - df["pickup_i"]) * df["progress_ratio"]
        ).astype(int)
        df["req_j"] = np.round(
            df["pickup_j"]
            + (df["dropoff_j"] - df["pickup_j"]) * df["progress_ratio"]
        ).astype(int)
        return df

    # -----------------------------------------------------------------
    def run(self, df_day: pd.DataFrame) -> pd.DataFrame:
        """
        執行完整 pipeline：
        初始電量分配 → 閾值篩選 → 時空內插 → 排序。
        """
        df = df_day.copy()
        df = self._assign_initial_battery(df)
        df = self._filter_triggered_evs(df)
        df = self._compute_request_location(df)
        df = df.sort_values("request_time").reset_index(drop=True)

        task_columns = [
            "request_time", "req_i", "req_j",
            "initial_kwh", "trip_dist_km",
        ]
        return df[task_columns]


# =====================================================================
# 5. MLDatasetGenerator — 將任務轉換為時空矩陣並切分
# =====================================================================
class MLDatasetGenerator:
    """將離散的 charging_tasks 轉換為 3D 時空矩陣，並進行 Train/Val/Test 依序切分。"""

    def __init__(self, config: GridConfig, time_interval_min: int = 30, seq_len: int = 6) -> None:
        self.cfg = config
        self.interval = f"{time_interval_min}min" # 每幀時間 (預設30分)
        self.seq_len = seq_len                    # 用過去 6 幀預測下 1 幀

    # -----------------------------------------------------------------
    def generate_tensors(self, df_tasks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """聚合空間與時間，產出 ConvLSTM 可讀取的滑動窗口特徵 (X) 與目標 (Y)。"""
        print(f"\n[ML Data Prep] 1. 轉換任務清單為時空網格影片 (每幀 {self.interval})...")
        df = df_tasks.copy()
        
        # 將發生時間無條件捨去到最近的 Time Bin
        df['time_bin'] = df['request_time'].dt.floor(self.interval)

        # 建立涵蓋整個月的完整時間軸，確保沒有漏掉任何時段
        time_bins = pd.date_range(
            start=self.cfg.time_start,
            end=self.cfg.time_end,
            freq=self.interval
        )
        num_steps = len(time_bins)

        # 初始化 [時間步, H, W] 的全零 Numpy 矩陣
        video_tensor = np.zeros((num_steps, self.cfg.grid_cells, self.cfg.grid_cells), dtype=np.float32)
        time_to_idx = {t: i for i, t in enumerate(time_bins)}

        # 統計每個網格在每個時間段的需求量
        grouped = df.groupby(['time_bin', 'req_i', 'req_j']).size().reset_index(name='demand')

        # 填入真實需求資料
        for _, row in grouped.iterrows():
            if row['time_bin'] in time_to_idx:
                t_idx = time_to_idx[row['time_bin']]
                i, j = int(row['req_i']), int(row['req_j'])
                if 0 <= i < self.cfg.grid_cells and 0 <= j < self.cfg.grid_cells:
                    video_tensor[t_idx, i, j] += row['demand']

        print(f"   => 原始時空矩陣聚合完成，形狀: {video_tensor.shape} (TimeSteps, W, H)")

        # 製作滑動窗口資料集 (Sliding Windows)
        print("[ML Data Prep] 2. 製作滑動窗口特徵矩陣...")
        X, Y = [], []
        for t in range(num_steps - self.seq_len):
            # X shape: (seq_len, 1通道, 100, 100)
            x_win = np.expand_dims(video_tensor[t : t + self.seq_len], axis=1)
            # Y shape: (1通道, 100, 100) -> 這是模型要預測的目標
            y_tgt = np.expand_dims(video_tensor[t + self.seq_len], axis=0)
            
            X.append(x_win)
            Y.append(y_tgt)

        return np.array(X), np.array(Y)

    # -----------------------------------------------------------------
    def split_and_save(self, X: np.ndarray, Y: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.2):
        """依時間序列嚴格切分 Train / Val / Test，並匯出為 .npy 檔案。"""
        total_samples = len(X)
        print(f"\n[ML Data Prep] 3. 依序切分訓練、驗證與測試集 (總樣本數: {total_samples})")

        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))

        # 嚴格時序切分 (防範資料穿越)
        X_train, Y_train = X[:train_end], Y[:train_end]
        X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
        X_test, Y_test = X[val_end:], Y[val_end:]

        print(f"   -> Train Size: {X_train.shape[0]} samples (前 {train_ratio*100:.0f}%, 供 Colab 訓練)")
        print(f"   -> Val Size:   {X_val.shape[0]} samples (中間 {val_ratio*100:.0f}%, 供 Colab 驗證)")
        print(f"   -> Test Size:  {X_test.shape[0]} samples (最後剩餘, 供 ALNS 模擬測試)")

        print("[ML Data Prep] 4. 正在寫入 .npy 矩陣檔案至硬碟...")
        np.save("X_train.npy", X_train)
        np.save("Y_train.npy", Y_train)
        np.save("X_val.npy", X_val)
        np.save("Y_val.npy", Y_val)
        np.save("X_test.npy", X_test)
        np.save("Y_test.npy", Y_test)
        print("✅ 所有矩陣儲存成功！準備上傳至 Google Drive。")

def visualize_spatiotemporal_heatmaps(df_tasks: pd.DataFrame, grid_size: int = 100, save_path: str = "charging_heatmaps.png"):
    """
    將充電請求資料依照 6 個時間區段切分，並繪製成 2x3 的平滑熱力圖。
    """
    print("🎨 正在繪製時空熱力圖 (Spatio-temporal Heatmaps)...")
    
    # 定義 6 個時間窗與對應的標題 (每 4 小時一張圖)
    time_windows = [
        (0, 4, "(a) 00:00-04:00 AM"),
        (4, 8, "(b) 04:00-08:00 AM"),
        (8, 12, "(c) 08:00-12:00 AM"),
        (12, 16, "(d) 12:00-16:00 PM"),
        (16, 20, "(e) 16:00-20:00 PM"),
        (20, 24, "(f) 20:00-24:00 PM")
    ]

    # 建立 2 行 3 列的畫布
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # 確保 request_time 是 datetime 格式，並提取「小時」
    df = df_tasks.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['request_time']):
        df['request_time'] = pd.to_datetime(df['request_time'])
        
    df['hour'] = df['request_time'].dt.hour

    for idx, (start_hour, end_hour, title) in enumerate(time_windows):
        ax = axes[idx]

        # 1. 篩選該時間段的資料
        mask = (df['hour'] >= start_hour) & (df['hour'] < end_hour)
        df_subset = df[mask]

        # 2. 建立 100x100 的二維直方圖 (計算每個網格的需求數量)
        heatmap, xedges, yedges = np.histogram2d(
            df_subset['req_i'],
            df_subset['req_j'],
            bins=grid_size,
            range=[[0, grid_size], [0, grid_size]]
        )

        # 3. 高斯濾波器平滑矩陣 (sigma 控制熱力圖擴散程度)
        heatmap_smoothed = gaussian_filter(heatmap, sigma=2.5)

        # 4. 繪製熱力圖 (使用 jet 或是 turbo 色譜)
        im = ax.imshow(heatmap_smoothed.T, origin='lower', cmap='jet', interpolation='nearest')

        # 5. 美化圖表
        ax.set_title(title, fontsize=16, pad=10)
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        
        # 隱藏 XY 軸的刻度，讓它更像真實的熱力分布地圖
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    # 調整整體排版並加上總標題
    plt.suptitle("Spatio-temporal Distribution of EV Charging Requests", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 儲存為高解析度圖片供論文使用
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 注意：由於使用了 'Agg' 後端，不要呼叫 plt.show()
    plt.close(fig) 
    print(f"✅ 熱力圖已成功生成並儲存為 {save_path}！")

# =====================================================================
# 6. Entry point (主程式執行區)
# =====================================================================
if __name__ == "__main__":
    config = GridConfig()

    # Stage 1: 旅程資料清洗與網格化
    processor = NYCTaxiGridProcessor(config)
    df_grid = processor.run()
    
    # Stage 2: 充電請求產生
    generator = ChargingRequestGenerator(config)
    charging_tasks = generator.run(df_grid)
    
    # Stage 3: ML 資料聚合與切分 (準備給 ConvLSTM)
    ml_prep = MLDatasetGenerator(config, time_interval_min=30, seq_len=6)
    X_all, Y_all = ml_prep.generate_tensors(charging_tasks)
    ml_prep.split_and_save(X_all, Y_all)
    config = GridConfig()

    # Stage 1: 旅程資料清洗與網格化
    processor = NYCTaxiGridProcessor(config)
    df_grid = processor.run()
    print(df_grid.head())

    # Stage 2: 充電請求產生
    generator = ChargingRequestGenerator(config)
    charging_tasks = generator.run(df_grid)
    print(charging_tasks.head())

    visualize_spatiotemporal_heatmaps(charging_tasks, grid_size=config.grid_cells)