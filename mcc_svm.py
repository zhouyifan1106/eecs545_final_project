#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:07:03 2018

@author: TLuv
"""

import numpy as np
from numpy import genfromtxt
from scipy.sparse import hstack
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



def convert_X(mat):
    mat = csr_matrix(mat)
    return mat

def fit(X, y,threshold=0.2):
    X_extended = X
    classifiers = []
    indices = []
    for i in range(y.shape[1]):
        classifier = SVC()#class_weight = 'balanced')
        added_y = []
        if i >= 1:
            for j in range(i):
                corr = np.corrcoef(y[:,j],y[:,i])[0][1]
                if corr >= threshold or corr <= -threshold:
                    added_y.append(j)
        if len(added_y) > 0:
            classifier.fit(np.hstack([X_extended,(y[:,added_y])]), y[:,i])
        else:
            classifier.fit(X_extended, y[:,i])
        classifiers.append(classifier)
        indices.append(added_y)
        print (i)
    return classifiers,indices

def predict(classifiers, indices, X):
    X_extended = X
    predictions = []
    for i in range(len(indices)):
        X_original = X_extended.copy()
        classifier = classifiers[i]
        index = indices[i]
        if len(index) >= 1:
            for ii in index:
                X_extended = np.hstack([X_extended,predictions[ii].reshape(X.shape[0],1)])
            pred = np.array(classifier.predict(X_extended)).T
        else:
            pred = np.array(classifier.predict(X_extended)).T
        predictions.append(pred)
        X_extended = X_original
        print (i)
    return predictions














