import numpy as np
import csv
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def kmeans(x, k, no_of_iterations):

    idx = np.random.choice(len(x), k)
    centroids = x[idx, :]

    distances = cdist(x, centroids, "euclidean")

    points = np.array([np.argmin(i) for i in distances])

    for _ in range(no_of_iterations):
        centroids = []
        for idx in range(k):
            temp_cent = x[points == idx].mean(axis=0)
            centroids.append(temp_cent)

        centroids = np.vstack(centroids)

        distances = cdist(x, centroids, "euclidean")
        points = np.array([np.argmin(i) for i in distances])
    return points


with open("hayes_roth.data") as f:
    lines = csv.reader(f)
    _dataset = list(lines)

dataset = np.array(_dataset).astype(np.float64)


points = kmeans(dataset, 3, 20)

cluster = {"0": 0, "1": 0, "2": 0}

for point in points:
    cluster[str(point)] += 1
print(cluster)