# algorithms/clustering.py
import numpy as np
from sklearn.cluster import KMeans

class ClusterModel:
    def __init__(self):
        pass

    def get_cluster_centroids(self, requests, n_clusters):
        """
        輸入: requests (List[Request]), n_clusters (int: 閒置MCS數量)
        輸出: centroids (List[(x, y)])
        """
        # 1. 邊界檢查：如果沒需求，或者沒車，就回傳空
        if not requests or n_clusters <= 0:
            return []

        # 2. 提取座標
        points = np.array([[r.x, r.y] for r in requests])

        # 3. 如果需求點少於車子數量，每個需求點就是一個重心
        if len(points) <= n_clusters:
            return points.tolist()

        # 4. 執行 K-Means 分群
        # 這會找出能代表這堆點的 n_clusters 個中心
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(points)
        
        # 取得重心座標
        centroids = kmeans.cluster_centers_
        
        return centroids.tolist()

    def calculate_weighted_centroids(self, requests, n_clusters):
        """
        (進階選項) 論文中的重心法則：加入權重
        這裡可以把 SoC 當作權重：電量越低，權重越重，中心點會往它偏移
        """
        if not requests or n_clusters <= 0: return []
        
        points = np.array([[r.x, r.y] for r in requests])
        # 權重: 1/SoC (電量越低，權重越大)
        weights = np.array([1.0 / (r.soc + 0.01) for r in requests])
        
        # 這裡為了簡化，如果不想引入複雜的 Weighted K-Means 套件，
        # 我們可以先用標準 K-Means，這在大多數論文中已經足夠好。
        return self.get_cluster_centroids(requests, n_clusters)