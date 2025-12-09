import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# trains a Random Forest
def train(x_train, y_train, max_depth=None, criterion='gini', n_estimators = 100):
    model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, 
                                   max_depth=max_depth, random_state=42, n_jobs=-1)
    
    model.fit(x_train, y_train)

    return model

# evaluate Random Forest
def evaluate(x_test, y_test, model):
    predictions = model.predict(x_test)
    
    return accuracy_score(y_test, predictions)

# search over 3 depths and a number of estimators to find the best parameters
def grid_search(x_val, y_val, x_train, y_train, depth_values, est_values, criteria="gini"):
    scores = {}
    best_depth = None
    best_acc = -1.0
    best_est = None

    for depth in depth_values:
        for est in est_values:
            print(f"Training RF tree with depth = {depth}")
            model = train(x_train, y_train, max_depth=depth, criterion = criteria, n_estimators=est)
            acc = evaluate(x_val, y_val, model)
            scores[(depth, est)] = acc

            print(f"val accuracy = {acc}")

            if acc > best_acc:
                best_acc = acc
                best_depth = depth
                best_est = est

    print(f"best parameters: depth {best_depth}, val accuracy = {best_acc}")
    
    return best_depth, best_est, best_acc, scores
