import argparse
import numpy as np
import dt_breed

from feature_extractor import load_features
from baselines import evaluate_baseline
import knn_breed

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

    # k_values = [1, 3, 5, 7, 11, 15]

    # scores = knn_breed.grid_search(x_test[:300], y_test[:300], x_train[:2000], y_train[:2000], k_values)
    # scores = knn_breed.grid_search(x_test, y_test, x_train[:2000], y_train[:2000], k_values)

    # for k, acc in scores.items():
    #     print(f"Accuracy for k = {k} is {acc}")

    depth_values = [5, 10, 20]
    criteria = "gini"


    best_depth, best_val_acc, scores = dt_breed.grid_search(x_val[:300], y_val[:300], x_train[:2000], y_train[:2000], depth_values, criteria)

    print("before best_model")

    best_model = dt_breed.train(x_train[:2000], y_train[:2000], max_depth=best_depth, criterion=criteria)

    for depth, acc in scores.items():
        print(f"depth = {depth}, accuracy = {acc}")

    print(f"Best parameters: depth = {best_depth}") 
    print(f"Best validation accuracy = {best_val_acc}")

    test_acc = dt_breed.evaluate(x_test, y_test, best_model)

    print(f"Decision Tree test accuracy: {test_acc}")

if __name__ == "__main__":
    main()