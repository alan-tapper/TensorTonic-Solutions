import numpy as np

def gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p ** 2)

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    n, d = X.shape

    best_score, best_split = np.inf, None
    for f in range(d):
        vals = np.unique(X[:, f])
        thresholds = (vals[:-1] + vals[1:]) / 2
        for t in thresholds:
            left = y[X[:, f] <= t]
            right = y[X[:, f] > t]
            
            score = (len(left) * gini(left) + len(right) * gini(right)) / n
            if score < best_score:
                best_score = score
                best_split = [f, float(t)]
    return best_split