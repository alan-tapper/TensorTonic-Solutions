import numpy as np
import math
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    points = np.asarray(points, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    diffs = points[:, None, :] - centroids[None, :, :]

    distances = np.sum(diffs ** 2, axis=2)
    print(distances)
    return np.argmin(distances, axis=1).tolist()

    
    # P = len(points)
    # C = len(centroids)
    # distances = [[] for p in range(P)]

    # for i in range(P):
    #     p = points[i]
    #     for j in range(C):
    #         c = centroids[j]
    #         distances[i].append(math.sqrt((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2))
    # distances = np.asarray(distances, dtype=float)
    # # print(distances)
    # assignments = np.argmin(distances, axis=1)
    # print(assignments)
    # return assignments.tolist()