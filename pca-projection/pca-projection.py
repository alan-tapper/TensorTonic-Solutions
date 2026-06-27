import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.asarray(X, dtype=float)
    
    # center the data at 0
    X_c = X - np.mean(X, axis=0)

    #covariance matrix
    C = (X_c.T @ X_c) / (X.shape[0] - 1)

    #eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    order = np.argsort(eigenvalues)[::-1]      # descending
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]      # reorder COLUMNS, not rows
    print(eigenvectors, eigenvalues)
    W = eigenvectors[:, :k]
    print(W)
    X_proj = X_c @ W
    return X_proj