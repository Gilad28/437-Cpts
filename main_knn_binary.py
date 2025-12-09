import numpy as np
from feature_extractor import load_features
from baselines import evaluate_baseline
import knn_binary


def main():
    args = parser.parse_args()
    data = load_features("./features/oxford_pets_hog")
    
    x_train = data["X_train"]
    y_train = data["y_train"]
    x_val = data["X_val"]
    y_val = data["y_val"]
    x_test = data["X_test"]
    y_test = data["y_test"]

    evaluate_baseline(x_train, y_train, x_val, y_val, x_test, y_test, task_name="Binary dog vs non-dog classification (KNN)")

    k_values = [1, 3, 5, 7, 11, 20, 25]
    print("\nRunning KNN grid search...")
    scores = knn_binary.grid_search(x_val, y_val, x_train, y_train, k_values)

    best_k = max(scores, key=scores.get)
    best_val_acc = scores[best_k]

    print("before best_model")
    print("Grid search results:")
    for k in k_values:
        print(f"k = {k}, accuracy = {scores[k]}")

    print(f"Best parameters: k = {best_k}")
    print(f"Best validation accuracy = {best_val_acc}")

    test_acc = knn_binary.evaluate(x_test, y_test, x_train, y_train, best_k)
    print(f"KNN test accuracy: {test_acc}")


if __name__ == "__main__":
    main()
