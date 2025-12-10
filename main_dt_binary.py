import numpy as np
import dt_binary

from feature_extractor import load_features
from baselines import evaluate_baseline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def main():
    data = load_features("./features/oxford_pets_hog")
    
    x_train = data["X_train"]
    y_train = data["y_train"]
    x_val = data["X_val"]
    y_val = data["y_val"]
    x_test = data["X_test"]
    y_test = data["y_test"]

    evaluate_baseline(x_train, y_train, x_val, y_val, x_test, y_test, task_name="Binary dog vs non-dog classification (Decision Tree)")

    depth_values = [1, 2, 3, 4, 5, 10, 15, 20]
    criteria = "gini"

    best_depth, best_val_acc, scores = dt_binary.grid_search(x_val, y_val, x_train, y_train, depth_values, criteria)

    print("before best_model")

    best_model = dt_binary.train(x_train, y_train, max_depth=best_depth, criterion=criteria)

    for depth, acc in scores.items():
        print(f"depth = {depth}, accuracy = {acc}")

    print(f"Best parameters: depth = {best_depth}") 
    print(f"Best validation accuracy = {best_val_acc}")

    test_acc = dt_binary.evaluate(x_test, y_test, best_model)

    print(f"Decision Tree test accuracy: {test_acc}")

    # Classification report (precision, recall, f1)
    y_pred = best_model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification report (Decision Tree):")
    print(classification_report(y_test, y_pred))

    # Save report to CSV
    import csv
    with open("dt_binary_classification_report.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        # Header row
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
