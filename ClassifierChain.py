#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 00:29:51 2018

@author: TLuv
"""


import numpy as np
from scipy.sparse import hstack
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import utils as util


def convert_X(mat):
    mat = csr_matrix(mat)
    return mat

def fit(X, y):
    X_extended = X
    pca = util.pca(X_extended, 100)
    PCAs = []
    PCAs.append(pca)
    X_extended_pca = pca.transform(scale(X_extended))
    classifiers = []
    for i in range(y.shape[1]):
        classifier = LogisticRegression()
        classifier.fit(X_extended_pca, y[:,i])
        classifiers.append(classifier)
        X_extended = np.hstack([X_extended, (y[:,i]).reshape(X.shape[0],1)])
    return classifiers

def predict(classifiers, X):
    X_extended = X
    predictions = []
    for classifier in classifiers:
        pred = np.array(classifier.predict(X_extended)).T
        predictions.append(pred)
        X_extended = np.hstack([X_extended, pred.reshape(X.shape[0],1)])
    return predictions


#np.random.seed(0)
#X = np.random.randn(1000, 200)
#y = np.random.randn(1000, 20)
#y[y <= 0] = 0
#y[y > 0 ] = 1
#
#cc = fit(X, y)
#pred = predict(cc, X)
#pred2 = pred.toarray()



    
    





    