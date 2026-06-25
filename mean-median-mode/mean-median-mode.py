import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    mean = sum(x) / len(x)
    if len(x) % 2:
        x_sorted = sorted(x)
        median = x_sorted[len(x_sorted) // 2]
    else:
        x_sorted = sorted(x)
        median = .5 * (x_sorted[len(x_sorted) // 2] + x_sorted[len(x_sorted) // 2 - 1])
    freqs = {}
    for i in range(len(x)):
        if x[i] not in freqs:
            freqs[x[i]] = 1
        else:
            freqs[x[i]] += 1
    freqs_sorted = sorted(freqs.keys(), key=(lambda i: (-freqs[i], i)))
    mode = freqs_sorted[0]
    return mean, median, mode