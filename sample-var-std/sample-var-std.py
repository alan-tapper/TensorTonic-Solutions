import numpy as np
import math

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    n = len(x)
    mean = sum(x) / n
    deviations_sum = 0
    for i in range(n):
        deviations_sum += (x[i] - mean) ** 2
    variance = deviations_sum/(n-1)
    return variance, math.sqrt(variance)