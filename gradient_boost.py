import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier


X = np.load('X_train.npy')
y = np.load('y_train.npy')
X_new = np.load('X_test.npy')
y_new = np.load('y_test.npy')


def grid_search(X, y):
    clf = GradientBoostingClassifier()
    params = {'loss': ['deviance', 'exponential'],
              'learning_rate': [0.01, 0.1, 0.5],
              'n_estimators': [20, 50],
              'max_depth': [1, 3, 5],
              'min_samples_split': [2, 4],
              'min_samples_leaf': [2, 4],
              'max_features': [None, 2, 4],
              'max_leaf_nodes': [None, 5],
              'warm_start': [True, False],
              'random_state': [1]}

    gs = GridSearchCV(estimator=clf, param_grid=params, n_jobs=-1, verbose=True)
    gs.fit(X, y)
    best_model = gs.best_estimator_
    return best_model


def gb_train_predict():
    seed = 34
    test_size = 0.20

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # model = grid_search(X_train, y_train)
    model = GradientBoostingClassifier(criterion='friedman_mse', init=None, learning_rate=0.5, loss='deviance', max_depth=6, max_features=4, max_leaf_nodes=3, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=500, presort='auto', random_state=1, subsample=1.0, verbose=0, warm_start=True)

    gb_clf = model.fit(X_train, y_train)

    # some cross validation
    print("Cross validation:")
    accuracy_scores = cross_val_score(gb_clf, X_train, y_train, cv=5, scoring='accuracy')
    print("Accuracy: %.2f%%" % (np.mean(accuracy_scores) * 100.0))
    f1_scores = cross_val_score(gb_clf, X_train, y_train, cv=5, scoring='f1')
    print("f1: %.2f%%" % (np.mean(f1_scores) * 100.0))
    print("\n")

    # make predictions for training data
    y_pred = gb_clf.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    print("Model evaluation on training data:")
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    recall = recall_score(y_test, predictions)
    print("Recall: %.2f%%" % (recall * 100.0))
    precision = precision_score(y_test, predictions)
    print("Precision: %.2f%%" % (precision * 100.0))
    f1 = f1_score(y_test, predictions)
    print("f1: %.2f%%" % (f1 * 100.0))
    print("\n")

    # make predictions for new unseen data
    y_pred_new = gb_clf.predict(X_new)
    pred_new = [round(value) for value in y_pred_new]

    # evaluate unseen data predictions
    print("Model evaluation on unseen data:")
    accuracy = accuracy_score(y_new, pred_new)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    recall = recall_score(y_new, pred_new)
    print("Recall: %.2f%%" % (recall * 100.0))
    precision = precision_score(y_new, pred_new)
    print("Precision: %.2f%%" % (precision * 100.0))
    f1 = f1_score(y_new, pred_new)
    print("f1: %.2f%%" % (f1 * 100.0))


if __name__ == '__main__':
    gb_train_predict()
