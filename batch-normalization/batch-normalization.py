import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)
    
    if x.ndim == 2:
        #(N,D)
        mu = np.mean(x, axis=0)
        sigma_squared = np.mean((x - mu) ** 2, axis=0)
        x_hat = (x - mu) / np.sqrt(sigma_squared + eps)
        y = gamma * x_hat + beta
        return y
    else:
        #(N,C,H,W)
        mu = np.mean(x, axis=(0, 2, 3), keepdims=True)
        sigma_squared = np.mean((x - mu) ** 2, axis=(0, 2, 3), keepdims=True)
        x_hat = (x - mu) / np.sqrt(sigma_squared + eps)
        y = gamma.reshape(1, -1, 1, 1) * x_hat + beta.reshape(1, -1, 1, 1)
        return y