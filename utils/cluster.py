#python3.10
from sklearn.cluster import KMeans, AgglomerativeClustering

def cluster_x(data, n_clusters=None, method='agglomerative'):
    """
        cluster the feature tracks by the rigid part estimation
    Args:
        data: feature tracks    N C
        n_clusters: number of clusters
        method: clustering method
    """
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(distance_threshold=7*20,
                                        linkage='ward', n_clusters=None,
                                        metric='euclidean')
    else:
        raise ValueError('Invalid clustering method')
    
    return model.fit_predict(data)


