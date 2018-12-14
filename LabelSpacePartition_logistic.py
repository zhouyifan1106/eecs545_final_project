#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:24:33 2018

@author: naichen
"""

import LabelPowerset_logistic as lp_logistic
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import hstack


def fit(X, trans_result,cluster):
    classifiers = []
    for i in range(len(cluster)):
        classifier = lp_logistic.fit(X, trans_result[i])
        classifiers.append(classifier)
    return classifiers

#def convert_X(mat):
#    mat = csr_matrix(mat)
#    return mat

def predict(classifiers, X, trans_result):
    new_X = X    
    predictions = [csc_matrix(np.array(lp_logistic.predict(classifiers[i], new_X, trans_result[i]))) for i in range(len(classifiers))]
    return hstack(predictions)

