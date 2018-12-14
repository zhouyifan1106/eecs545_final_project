#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:38:59 2018

@author: TLuv
"""

import numpy as np
from scipy.sparse import hstack
from scipy.sparse import csr_matrix, csc_matrix
import copy
from sklearn.svm import SVC


###################################### Binary Relevance ########################################
def convert_X(mat):
    mat = csr_matrix(mat)
    return mat

def fit(X, y):
    classifiers = []
    for i in range(y.shape[1]):
        classifier = SVC()
        classifier.fit(X, y[:,i])
        classifiers.append(classifier)
    return classifiers

def predict(classifiers, X):
    predictions = [csc_matrix(np.array(classifier.predict(X))).T for classifier in classifiers]
    return hstack(predictions)