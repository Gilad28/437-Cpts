import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
    return best_depth, best_acc, scores
