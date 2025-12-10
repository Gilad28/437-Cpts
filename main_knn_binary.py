import numpy as np
from feature_extractor import load_features
from baselines import evaluate_baseline
import knn_binary
from sklearn.metrics import classification_report


def main():
    data = load_features("./features/oxford_pets_hog")
    
    x_train = data["X_train"]
    y_train = data["y_train"]
    x_val = data["X_val"]
    y_val = data["y_val"]
    x_test = data["X_test"]
    y_test = data["y_test"]

    evaluate_baseline(x_train, y_train, x_val, y_val, x_test, y_test, task_name="Binary dog vs non-dog classification (KNN)")

    k_values = [1, 3, 5, 7, 11, 15, 20, 30, 40, 50]
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

    # Classification report (precision, recall, f1)
    # Build predictions for the test set using best_k
    y_pred = []
    for i in range(x_test.shape[0]):
        y_pred.append(knn_binary.predict(x_test[i], x_train, y_train, best_k))
    y_pred = np.array(y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification report (KNN):")
    print(classification_report(y_test, y_pred))

    # Save report to CSV
    import csv
    with open("knn_binary_classification_report.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                writer.writerow([
                    label,
                    metrics.get("precision", ""),
                    metrics.get("recall", ""),
                    metrics.get("f1-score", ""),
                    metrics.get("support", "")
                ])


if __name__ == "__main__":
    main()
