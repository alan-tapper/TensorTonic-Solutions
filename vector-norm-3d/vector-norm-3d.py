import numpy as np
import math

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    # Your code here
    v = np.asarray(v, dtype=float)
    squared = v ** 2
    summed = np.sum(v**2, axis=-1)
    return np.sqrt(summed)