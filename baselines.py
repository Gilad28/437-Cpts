import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# simple baseline classifiers for comparison

class MajorityClassifier:
    # just always predicts the most common class
    def __init__(self):
        self.majority_class = None
    
    def fit(self, X, y):
        counter = Counter(y)
        self.majority_class = counter.most_common(1)[0][0]
        return self
    
    def predict(self, X):
        # return array of same class for everything
        return np.full(len(X), self.majority_class)
    
    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)


class RandomClassifier:
    # randomly guess classes (uniform probability)
    def __init__(self, random_state=42):
        self.classes = None
        self.random_state = random_state
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        return self
    
    def predict(self, X):
        np.random.seed(self.random_state)
        return np.random.choice(self.classes, size=len(X))
    
    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)


def evaluate_baseline(X_train, y_train, X_val, y_val, X_test, y_test, 
                      task_name="Classification"):
    # runs both baseline classifiers and prints results
    print("\n" + "=" * 60)
    print(f"BASELINE EVALUATION: {task_name}")
    print("=" * 60)
    
    # show class distribution first
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nClass distribution in training set:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    # test majority class baseline
    print("\n--- Majority Class Baseline ---")
    maj_clf = MajorityClassifier()
    maj_clf.fit(X_train, y_train)
    
    train_acc = maj_clf.score(X_train, y_train)
    val_acc = maj_clf.score(X_val, y_val)
    test_acc = maj_clf.score(X_test, y_test)
    
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Always predicts class: {maj_clf.majority_class}")
    
    # test random baseline
    print("\n--- Random Baseline ---")
    rand_clf = RandomClassifier()
    rand_clf.fit(X_train, y_train)
    
    train_acc = rand_clf.score(X_train, y_train)
    val_acc = rand_clf.score(X_val, y_val)
    test_acc = rand_clf.score(X_test, y_test)
    
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Expected accuracy for {len(unique)} classes: ~{1/len(unique):.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    # quick test with fake data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                               weights=[0.7, 0.3], random_state=42)
    
    # split it up
    X_train, X_test = X[:700], X[700:850]
    X_val = X[850:]
    y_train, y_test = y[:700], y[700:850]
    y_val = y[850:]
    
    evaluate_baseline(X_train, y_train, X_val, y_val, X_test, y_test, 
                     task_name="Binary Classification Example")

