# algorithms/clustering.py
import numpy as np
from sklearn.cluster import KMeans

class ClusterModel:
    def __init__(self):
        pass

    def get_cluster_centroids(self, requests, n_clusters):
        """
        輸入: requests (List[Request]), n_clusters (int: 欲分群數量)
        輸出: centroids (List[(x, y)])
        """
        if not requests:
            return []
        
        # 安全檢查：如果 n_clusters <= 0，設為 1 (至少算出一個重心)
        n_clusters = max(1, int(n_clusters))

        points = np.array([[r.x, r.y] for r in requests])

        # 如果點的數量少於要分的群數，直接回傳點本身即可
        if len(points) <= n_clusters:
            return points.tolist()

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(points)
        
        return kmeans.cluster_centers_.tolist()