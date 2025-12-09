import numpy as np
import queue
from collections import Counter

# KNN for binary dog vs non-dog classification

# Computes the Euclidean distance between a and b
def distance(a, b):
    return np.linalg.norm(a - b)


# Updates neighbors and distance queue (max-heap behavior via negative distances)
def _update(neighbors_and_distance, x_index, proximity, k=10):
    if neighbors_and_distance.qsize() < k:
        neighbors_and_distance.put((-proximity, x_index))
    else:
        (dist, _) = neighbors_and_distance.queue[0]
        if np.fabs(dist) > np.fabs(proximity):
            neighbors_and_distance.get()
            neighbors_and_distance.put((-proximity, x_index))
    return neighbors_and_distance


# Finds indices of k nearest neighbors
def find_nearest_neighbor(x_test, X, k=10):
    neighbors_and_distance = queue.PriorityQueue()

    for i in range(X.shape[0]):
        proximity = distance(x_test, X[i])
        neighbors_and_distance = _update(neighbors_and_distance, i, proximity, k=k)

    neighbors = []
    while not neighbors_and_distance.empty():
        (_, i) = neighbors_and_distance.get()
        neighbors.append(i)

    return neighbors


# Predicts binary label by majority vote among neighbors
def predict(x_test, X, Y, k=10):
    neighbors = find_nearest_neighbor(x_test, X, k=k)
    labels = [Y[i] for i in neighbors]
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


# Evaluate accuracy over a test set
def evaluate(X_tst, Y_tst, X_tr, Y_tr, k):
    right = 0
    for i in range(X_tst.shape[0]):
        if predict(X_tst[i], X_tr, Y_tr, k) == Y_tst[i]:
            right += 1
    return float(right) / float(X_tst.shape[0])


# Grid search over multiple k values; prints validation accuracies
def grid_search(x_val, y_val, X, Y, k_values):
    scores = {}
    for k in k_values:
        acc = evaluate(x_val, y_val, X, Y, k)
        scores[k] = acc
        print(f"Validation accuracy for k = {k}: {acc}")
    return scores
