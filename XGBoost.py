from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

X = np.load('X_train.npy')
y = np.load('y_train.npy')
X_new = np.load('X_test.npy')
y_new = np.load('y_test.npy')

def xg_train_predict():
    seed = 34
    test_size = 0.20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # fit model no training data
    model = XGBClassifier()
    bst = model.fit(X_train, y_train)

    # some cross validation
    f1_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

    print("f1_score:", f1_scores)
    accuracy_scores = cross_val_score(bst, X_train, y_train, cv=5, scoring='accuracy')
    print("accuracy:", accuracy_scores)


    # make predictions for test data
    y_pred = bst.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    recall = recall_score(y_test, predictions)
    print("Recall: %.2f%%" % (recall * 100.0))
    precision = precision_score(y_test, predictions)
    print("Precision: %.2f%%" % (precision * 100.0))
    f1 = f1_score(y_test, predictions)
    print("f1: ",f1)

    # make predictions for new unseen data
    y_pred_new = bst.predict(X_new)
    pred_new = [round(value) for value in y_pred_new]

    # evaluate unseen data predictions
    accuracy = accuracy_score(y_new, pred_new)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    recall = recall_score(y_new, pred_new)
    print("Recall: %.2f%%" % (recall * 100.0))
    precision = precision_score(y_new, pred_new)
    print("Precision: %.2f%%" % (precision * 100.0))
    f1 = f1_score(y_new, pred_new)
    print("f1: ",f1)
    return bst

if __name__ == '__main__':
    bst = xg_train_predict()
