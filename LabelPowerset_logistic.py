#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:16:23 2018

@author: naichen
"""

import numpy as np
#from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression



def transform(y):
        result1 = {}
        unique_combinations_ = {}
        reverse_combinations_ = []
        last_id = 0
        train_vector = []
        for i in range(y.shape[0]):
            labels_applied = y[i,:]
            label_string = ",".join(map(str, labels_applied))

            if label_string not in unique_combinations_:
                unique_combinations_[label_string] = last_id
                reverse_combinations_.append(labels_applied)
                last_id += 1

            train_vector.append(unique_combinations_[label_string])
        
        result1['unique_combinations_'] = unique_combinations_
        result1['reverse_combinations_'] = reverse_combinations_
        result1['train_vector'] = np.array(train_vector)
        result1['label_count'] = y.shape[1]
        return result1


def inverse_transform(y_trans, trans_result):
        label_count = trans_result['label_count']
        n_samples = len(y_trans)
        result = np.zeros((n_samples,label_count),dtype=int)
        for row in range(n_samples):
            assignment = y_trans[row]
            result[row,:] = trans_result['reverse_combinations_'][assignment]
        return result


#def convert_X(mat):
#    mat = csr_matrix(mat)
#    return mat


def fit(X, trans_result):
    classifier = LogisticRegression()
    fit_result = classifier.fit(X,trans_result['train_vector'])
    return fit_result


def predict(classifier, X, trans_result):
    pred = classifier.predict(X)
    pred_inverse = inverse_transform(pred,trans_result)
    return pred_inverse



