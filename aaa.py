import pandas as pd
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================
# 1. 讀取數據與地理圖資
# ==========================================
print("讀取資料中...")
# 保留了時間欄位 tpep_pickup_datetime
columns_to_keep = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID']
df = pd.read_parquet('yellow_tripdata_2025-01.parquet', columns=columns_to_keep)

zones = gpd.read_file('taxi_zones/taxi_zones.shp').to_crs("EPSG:4326")
zones['lon'] = zones.geometry.centroid.x
zones['lat'] = zones.geometry.centroid.y

# ==========================================
# 2. 合併起點與終點座標 (Mapping)
# ==========================================
print("映射起迄點座標...")
# 合併起點
df_od = df.merge(
    zones[['LocationID', 'lon', 'lat', 'zone']],
    left_on='PULocationID', right_on='LocationID', how='left'
).rename(columns={'lon': 'pickup_lon', 'lat': 'pickup_lat', 'zone': 'pickup_zone'}).drop(columns=['LocationID'])

# 合併終點
df_od = df_od.merge(
    zones[['LocationID', 'lon', 'lat', 'zone']],
    left_on='DOLocationID', right_on='LocationID', how='left'
).rename(columns={'lon': 'dropoff_lon', 'lat': 'dropoff_lat', 'zone': 'dropoff_zone'}).drop(columns=['LocationID'])

# 確保上車時間為 datetime 格式，才能以時間切片
df_od['tpep_pickup_datetime'] = pd.to_datetime(df_od['tpep_pickup_datetime'])

# ==========================================
# 3. 核心數據清洗 (Data Cleaning)
# ==========================================
print("執行數據清洗...")
initial_count = len(df_od)

# 清洗 A：移除缺少經緯度的遺失值 (NaN)
df_od = df_od.dropna(subset=['pickup_lon', 'dropoff_lon'])

# 清洗 B：移除 TLC 定義的「未知區域」 (LocationID 264, 265 是 Unknown)
invalid_zones = [264, 265]
df_od = df_od[~df_od['PULocationID'].isin(invalid_zones)]
df_od = df_od[~df_od['DOLocationID'].isin(invalid_zones)]

# 清洗 C：(可選) 移除「起點 = 終點」的區內短途行程，只保留跨區移動
df_od = df_od[df_od['PULocationID'] != df_od['DOLocationID']]

final_count = len(df_od)
print(f"清洗完成！移除了 {initial_count - final_count} 筆無效或區內移動紀錄。")


# ==========================================
# 4. 產出【時空動態】 O-D 流量矩陣 (Spatio-temporal Tensor)
# ==========================================
print("正在以 1 小時為單位聚合流量資料...")

# 【關鍵修改】：加入 pd.Grouper，按照每 1 小時 (freq='1h') 統計各路線流量
od_matrix = df_od.groupby([
    pd.Grouper(key='tpep_pickup_datetime', freq='1h'), # 時間維度 T
    'PULocationID', 'pickup_zone', 'pickup_lon', 'pickup_lat',
    'DOLocationID', 'dropoff_zone', 'dropoff_lon', 'dropoff_lat'
]).size().reset_index(name='trip_count')

# 按照「時間」先排序，再按照該時段的「流量大小」排序
od_matrix = od_matrix.sort_values(
    by=['tpep_pickup_datetime', 'trip_count'], 
    ascending=[True, False]
).reset_index(drop=True)

print("\n==========================================")
print("時空動態 O-D 矩陣預覽 (加入時間維度)：")
print("==========================================")
# 強制 Pandas 顯示所有欄位
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(od_matrix.head(10))

# 儲存包含時間的 CSV 檔案 (檔案會比靜態的更大)
od_matrix.to_csv('nyc_od_matrix_temporal.csv', index=False)
print("\n已成功儲存為 nyc_od_matrix_temporal.csv！")