import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    X_train, X_test = np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)      
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1) 

    diffs = X_test[:, None, :] - X_train[None, :, :]  
    distances = np.sum(diffs ** 2, axis=2)            
    nearest_neighbors = np.argsort(distances, axis=1)[:, :k]  

    if k > len(X_train):
        pad = np.full((len(X_test), k - len(X_train)), -1, dtype=int)
        nearest_neighbors = np.concatenate([nearest_neighbors, pad], axis=1)
        
    return nearest_neighbors