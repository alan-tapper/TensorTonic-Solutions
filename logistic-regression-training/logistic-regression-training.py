import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N = y.shape[0]
    w = np.zeros(X.shape[1])
    b = 0.0
    for i in range(steps):
        z = X @ w + b
        p = _sigmoid(z)

        grad_z = p - y
        grad_w = (X.T @ grad_z) / N
        grad_b = np.mean(grad_z)
        w -= grad_w * lr
        b -= grad_b * lr
    return w, b