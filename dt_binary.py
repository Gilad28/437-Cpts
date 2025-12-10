import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import csv
import matplotlib.pyplot as plt

# train decision tree
def train(x_train, y_train, max_depth=None, criterion='gini', random_state=42):
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)
    
    model.fit(x_train, y_train)
    return model

# evaluate decision tree
def evaluate(x_test, y_test, model):
    predictions = model.predict(x_test)
    
    return accuracy_score(y_test, predictions)

# grid search
def grid_search(x_val, y_val, x_train, y_train, depth_values, criteria='gini'):
    scores = {}
    best_depth = None
    best_acc = -1.0

    for depth in depth_values:
        print(f"Training decision tree with depth = {depth}")
        model = train(x_train, y_train, max_depth=depth, criterion=criteria)
        acc = evaluate(x_val, y_val, model)
        scores[depth] = acc

        print(f"val accuracy = {acc}")

        if acc > best_acc:
            best_acc = acc
            best_depth = depth

    print(f"best parameters: depth {best_depth}, val accuracy = {best_acc}")

    with open("dt_binary_depth_results.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["max_depth", "validation_accuracy"])
        for d in depth_values:
            writer.writerow([d, scores[d]])

    plt.figure(figsize=(8, 5))
    plt.plot(depth_values, [scores[d] for d in depth_values], marker="o")
    plt.xlabel("Max Depth")
    plt.ylabel("Validation Accuracy")
    plt.title("Decision Tree: Depth vs Validation Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("dt_binary_depth_vs_accuracy.png")

    return best_depth, best_acc, scores
