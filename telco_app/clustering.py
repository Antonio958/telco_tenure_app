
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Clusterer:
    def __init__(self, k_min=2, k_max=8, random_state=42):
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state

    def best_k_silhouette(self, X):
        best_k, best_sil = None, -1
        for k in range(self.k_min, self.k_max + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k
        return best_k, best_sil
