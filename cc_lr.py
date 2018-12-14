import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def fit(X, y):
    classifiers = []
    for i in range(y.shape[1]):
        classifier = LogisticRegression()
        classifier.fit(X, y[:,i])
        classifiers.append(classifier)
        X = np.hstack([X, (y[:,i]).reshape(X.shape[0],1)])
    return classifiers

def predict(classifiers, X):
    predictions = []
    for classifier in classifiers:
        pred = np.array(classifier.predict(X)).T
        predictions.append(pred)
        X = np.hstack([X, pred.reshape(X.shape[0],1)])
    return predictions

    
    





    