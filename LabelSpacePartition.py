#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:14:19 2018

@author: naichen
"""
import LabelPowerset as lp
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import hstack

def fit(X, trans_result,cluster):
    classifiers = []
    for i in range(len(cluster)):
        classifier = lp.fit(X, trans_result[i])
        classifiers.append(classifier)
    return classifiers

#def convert_X(mat):
#    mat = csr_matrix(mat)
#    return mat

def predict(classifiers, X, trans_result):
    new_X = X    
    predictions = [csc_matrix(np.array(lp.predict(classifiers[i], new_X, trans_result[i]))) for i in range(len(classifiers))]
    return hstack(predictions)

