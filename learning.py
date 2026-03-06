import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# ==========================================
# 1. 讀取資料與特徵工程 (Feature Engineering)
# ==========================================
print("正在讀取資料與建立特徵...")
df = pd.read_csv('nyc_od_matrix.csv')

# 定義計算地球表面兩點距離的函數 (Haversine Formula)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # 地球半徑 (公里)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# 建立新特徵：直線距離與經緯度位移
df['distance_km'] = haversine(df['pickup_lat'], df['pickup_lon'], df['dropoff_lat'], df['dropoff_lon'])
df['delta_lon'] = df['dropoff_lon'] - df['pickup_lon']
df['delta_lat'] = df['dropoff_lat'] - df['pickup_lat']

# ==========================================
# 2. 定義特徵 (X) 與 預測目標 (y)，並切割資料集
# ==========================================
print("正在切割訓練集與測試集...")
features = ['pickup_lon', 'pickup_lat', 'dropoff_lon', 'dropoff_lat', 'distance_km', 'delta_lon', 'delta_lat']
target = 'trip_count'

X = df[features]
y = df[target]

# 將資料切分為 80% 訓練集，20% 測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. 訓練 XGBoost 機器學習模型
# ==========================================
print("正在訓練 XGBoost 模型 (這可能需要幾秒鐘)...")
# 設定 XGBoost 參數
model = xgb.XGBRegressor(
    n_estimators=200,     # 樹的數量
    learning_rate=0.05,   # 學習率
    max_depth=6,          # 樹的最大深度
    random_state=42
)

# 讓模型從訓練集 (X_train, y_train) 中學習規則
model.fit(X_train, y_train)

# ==========================================
# 4. 模型評估 (預測未知的測試集)
# ==========================================
print("正在評估模型表現...")
# 用訓練好的模型去預測測試集的流量
predictions = model.predict(X_test)

# 計算預測誤差
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n================ 模型評估結果 ================")
print(f"R-squared (R2 Score): {r2:.4f} (越接近 1 越好，代表模型能解釋多少比例的流量變化)")
print(f"MAE (平均絕對誤差):   {mae:.2f} (模型預測的趟數，平均跟真實答案差了幾趟)")
print(f"RMSE (均方根誤差):    {rmse:.2f} (對極端值比較敏感的誤差指標)")
print("==============================================")

# 顯示前 5 筆預測結果與真實結果的對比
results_df = pd.DataFrame({
    '真實流量 (Actual)': y_test.values[:5],
    'AI 預測流量 (Predicted)': np.round(predictions[:5]).astype(int)
})
print("\n[預測範例 - 前 5 筆測試資料]")
print(results_df)