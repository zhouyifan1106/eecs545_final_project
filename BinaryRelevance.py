#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 18:16:04 2018

@author: TLuv
"""

import copy
import numpy as np
from scipy.sparse import hstack, issparse, lil_matrix
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

###################################### Binary Relevance ########################################
def convert_X(mat):
    mat = csr_matrix(mat)
    return mat

def fit(X, y):
    classifiers = []
    for i in range(y.shape[1]):
        classifier = LogisticRegression()
        classifier.fit(X, y[:,i])
        classifiers.append(classifier)
    return classifiers

def predict(classifiers, X):
    #new_X = convert_X(X)   
    predictions = [csc_matrix(np.array(classifier.predict(X))).T for classifier in classifiers]
    return hstack(predictions)


#np.random.seed(0)
#X = np.random.randn(1000, 200)
#y = np.random.randn(1000, 20)
#y[y <= 0] = 0
#y[y > 0 ] = 1
#
#br = fit(X, y)
#pred = predict(br, X)
#pred2 = pred.toarray()


    
    





    