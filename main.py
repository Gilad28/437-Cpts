import argparse
import numpy as np
import dt_breed

from feature_extractor import load_features
from baselines import evaluate_baseline
import knn_breed
import random_forest_breed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=str, default="./features/stanford_dogs_hog")
    parser.add_argument("--task-name", type=str, default="Dog breed classification")

    args = parser.parse_args()
    data = load_features(args.feature_dir)
    
    x_train = data["X_train"]
    y_train = data["y_train"]
    x_val = data["X_val"]
    y_val = data["y_val"]
    x_test = data["X_test"]
    y_test = data["y_test"]

    evaluate_baseline(x_train, y_train, x_val, y_val, x_test, y_test, task_name=args.task_name,)

    k_values = [1]
    metric = "manhattan"

    scores = knn_breed.grid_search(x_test[:1000], y_test[:1000], x_train[:9000], y_train[:9000], k_values, metric)
    # scores = knn_breed.grid_search(x_test, y_test, x_train[:2000], y_train[:2000], k_values)

    for k, acc in scores.items():
        print(f"Accuracy for k = {k} is {acc}")

    best_k = max(scores, key=scores.get)
    best_val = scores[best_k]

    print(f"Best k = {best_k}, best value = {best_val}")

    test_acc = knn_breed.evaluate(x_test, y_test, x_train[:4000], y_train[:4000], k = best_k, metric = metric)

    print(f"Test accuracy = {test_acc}")



    # # depth_values = [5, 10, 20]
    # # criteria = "gini"


    # # best_depth, best_val_acc, scores = dt_breed.grid_search(x_val[:300], y_val[:300], x_train[:2000], y_train[:2000], depth_values, criteria)

    # # print("before best_model")

    # # best_model = dt_breed.train(x_train[:2000], y_train[:2000], max_depth=best_depth, criterion=criteria)

    # # for depth, acc in scores.items():
    # #     print(f"depth = {depth}, accuracy = {acc}")

    # # print(f"Best parameters: depth = {best_depth}") 
    # # print(f"Best validation accuracy = {best_val_acc}")

    # # test_acc = dt_breed.evaluate(x_test, y_test, best_model)

    # # print(f"Decision Tree test accuracy: {test_acc}")




    # # depth_values = [5, 10, 20, None]
    # depth_values = [None]

    # criteria = "gini"
    # est_values = [600]

    # # best_depth, best_est, best_val_acc, scores = random_forest_breed.grid_search(x_val[:300], y_val[:300], x_train[:2000], y_train[:2000], 
    # #                                                                             depth_values, est_values, criteria)
    
    # best_depth, best_est, best_val_acc, scores = random_forest_breed.grid_search(x_val, y_val, x_train, y_train, 
    #                                                                             depth_values, est_values, criteria)


    # print("before best_model")

    # for params, acc in scores.items():
    #     print(f"depth = {params[0]}, estimators = {params[1]}, accuracy = {acc}")

    # print(f"Best parameters: depth = {best_depth}, estimators = {best_est}") 
    # print(f"Best validation accuracy = {best_val_acc}")

    # # best_model = random_forest_breed.train(x_train[:2000], y_train[:2000], max_depth=best_depth, criterion=criteria, n_estimators = best_est)

    # best_model = random_forest_breed.train(x_train, y_train, max_depth=best_depth, criterion=criteria, n_estimators = best_est)

    # test_acc = random_forest_breed.evaluate(x_test, y_test, best_model)

    # print(f"RF with test accuracy: {test_acc}")
    

if __name__ == "__main__":
    main()