import numpy as np
import queue
# import os
from collections import Counter

# KNN implementation for dog breed classfication.

# Returns the most frequent label among the neighbors
def dominant_label(neighbors, Y):

    labels = [Y[i] for i in neighbors]
    counts = Counter(labels)
    dominant_label = counts.most_common(1)[0][0]

    return dominant_label


# Computes the Euclidean distance between a and b
def distance(a, b):

    dist = np.linalg.norm(a - b)

    return dist

# Updates neighbors and distance queue
def update(neighbors_and_distance, x_index, proximity, k = 10):

    if neighbors_and_distance.qsize() < k:

        # Inserts the new neighbor with negative distance (max-heap behavior)
        neighbors_and_distance.put((-proximity, x_index))

    else:

        (dist, i) = neighbors_and_distance.queue[0]

        if np.fabs(dist) > np.fabs(proximity):
            # Removes the farthest neighbor and inserts the closer one
            neighbors_and_distance.get()
            neighbors_and_distance.put((-proximity, x_index))

    return neighbors_and_distance

# find the nearest neighbors and returns the queue
def find_nearest_neighbor(x_test, X, k = 10):

    neighbors_and_distance = queue.PriorityQueue()  # decide how to organize this such that it will work in tandem with the update function below

    for i in range(X.shape[0]):

        # Computes distance between x_test and x_candidate and updates the queue of k nearest neighbors
        proximity = distance(x_test, X[i])

        neighbors_and_distance = update(neighbors_and_distance, i, proximity, k = k)  # this needs to be implemented in tandem with how neighbors_and_distance is organized

    neighbors = []

    while not neighbors_and_distance.empty():

        (_, i) = neighbors_and_distance.get()
        neighbors.append(i)

    return neighbors

# predicts label
def predict(x_test, X, Y, k = 10):

    neighbors = []
    for i in range(X.shape[0]):
        proximity = distance(x_test, X[i])
        neighbors.append((proximity, Y[i]))


    neighbors = find_nearest_neighbor(x_test, X, k = k)  # return a list of indices

    return dominant_label(neighbors, Y)  # return the dominant label among the neighbor

# evaluate K-NN
def evaluate(X_tst, Y_tst, X_tr, Y_tr, k):

    # Return accuracy as number of correct predictions / total
    right = 0

    for i in range(X_tst.shape[0]):
        if predict(X_tst[i], X_tr, Y_tr, k) == Y_tst[i]:
            right += 1

    return float(right) / float(X_tst.shape[0])

# finds the validation accuracy of each k value
def grid_search(x_test, y_test, X, Y, k_values):
    scores = {}

    for k in k_values:
        acc = evaluate(x_test, y_test, X, Y, k)
        scores[k] = acc

    return scores
