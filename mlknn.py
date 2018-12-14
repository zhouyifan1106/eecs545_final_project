#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 02:53:08 2018

@author: TLuv
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

def _compute_prior(s, k, y):
        prior_prob_true = np.array((s + y.sum(axis=0)) / (s * 2 + y.shape[0]))[0]
        prior_prob_false = 1 - prior_prob_true

        return (prior_prob_true, prior_prob_false)
    
def _compute_cond(s, k, X, y):
        knn = NearestNeighbors(k).fit(X)
        c = sparse.lil_matrix((y.shape[1], k + 1), dtype='i8')
        cn = sparse.lil_matrix((y.shape[1], k + 1), dtype='i8')

        label_info = sparse.dok_matrix(y)

        neighbors = [a[0:] for a in
                     knn.kneighbors(X, k + 0, return_distance=False)]

        for instance in range(y.shape[0]):
            deltas = label_info[neighbors[instance], :].sum(axis=0)
            for label in range(y.shape[1]):
                if label_info[instance, label] == 1:
                    c[label, deltas[0, label]] += 1
                else:
                    cn[label, deltas[0, label]] += 1

        c_sum = c.sum(axis=1)
        cn_sum = cn.sum(axis=1)

        cond_prob_true = sparse.lil_matrix((y.shape[1], k + 1), dtype='float')
        cond_prob_false = sparse.lil_matrix((y.shape[1], k + 1), dtype='float')
        for label in range(y.shape[1]):
            for neighbor in range(k + 1):
                cond_prob_true[label, neighbor] = (s + c[label, neighbor]) / (
                        s * (k + 1) + c_sum[label, 0])
                cond_prob_false[label, neighbor] = (s + cn[label, neighbor]) / (
                        s * (k + 1) + cn_sum[label, 0])
        return knn, cond_prob_true, cond_prob_false
    
def predict(s, k, x_train, x_test, y_train):
    
        prior_prob_true, prior_prob_false = _compute_prior(s, k, sparse.lil_matrix(y_train))
        knn, cond_prob_true, cond_prob_false = _compute_cond(s, k, x_train, sparse.lil_matrix(y_train))

        result = sparse.lil_matrix((x_test.shape[0], y_train.shape[1]), dtype='i8')
        neighbors = [a[0:] for a in
                     knn.kneighbors(x_test, k + 0, return_distance=False)]
        for instance in range(x_test.shape[0]):
            deltas = sparse.lil_matrix(y_train)[neighbors[instance],].sum(axis=0)

            for label in range(y_train.shape[1]):
                p_true = prior_prob_true[label] * cond_prob_true[label, deltas[0, label]]
                p_false = prior_prob_false[label] * cond_prob_false[label, deltas[0, label]]
                result[instance, label] = int(p_true >= p_false)

        return result
    
    
    