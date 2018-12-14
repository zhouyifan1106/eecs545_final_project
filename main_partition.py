#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 21:03:45 2018

@author: naichen
"""

import numpy as np
import LabelSpacePartition_logistic as lsp_logistic
import LabelSpacePartition_svm as lsp_svm
import LabelPowerset_logistic as lp_logistic
import LabelPowerset_svm as lp_svm
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss



##——————————————————run the algorithm on simulated data————————————————————————
## import simulated data
X_train = np.genfromtxt('x.csv', delimiter = ',')
y_train = np.genfromtxt('y.csv', delimiter = ',')

X_train_train = X_train[0:7999, :]
X_train_test = X_train[8000:, :]
y_train_train = y_train[0:7999, :]
y_train_test = y_train[8000:, :]

y_trainframe = pd.DataFrame(y_train)
corr_matrix = y_trainframe.corr()
large_cor={}
for i in range(len(corr_matrix)):
    for j in range(i,len(corr_matrix)):
        if abs(corr_matrix.iloc[i,j])>=0.2 and corr_matrix.iloc[i,j]!=1.0:
            row_col = i,j
            large_cor[row_col]=corr_matrix.iloc[i,j]
candiadate= np.unique(np.array(list(large_cor.keys())).flatten())

cluster = [[0,1,3,5],[2,4]]      
y_train_train1 = y_train_train[:,np.array(cluster[0])]
y_train_train2 = y_train_train[:,np.array(cluster[1])]      
trans_result1 = lp_svm.transform(y_train_train1)
trans_result2 = lp_svm.transform(y_train_train2)
trans_result = [trans_result1]
trans_result.append(trans_result2)

# Fit using lp
lsp_fit = lsp_svm.fit(X_train_train,trans_result,cluster)
lsp_pred = lsp_svm.predict(lsp_fit ,X_train_test,trans_result).toarray()

y_train_test = y_train_test.astype(np.int64)
column_index = np.array([0,1,3,5,2,4])
y_train_test = y_train_test[:,column_index]

print('The accuracy score of simulated data is ', accuracy_score(lsp_pred,y_train_test))
print('The hamming loss of simulated data is ', 1-hamming_loss(lsp_pred,y_train_test))
 



##——————————————————run the algorithm on music data————————————————————————
## import music data
X_train = np.genfromtxt('x_music.csv', delimiter = ',')
y_train = np.genfromtxt('y_music.csv', delimiter = ',')

X_train_train = X_train[0:499, :]
X_train_test = X_train[500:, :]
y_train_train = y_train[0:499, :]
y_train_test = y_train[500:, :]

y_trainframe = pd.DataFrame(y_train)
corr_matrix = y_trainframe.corr()
large_cor={}
for i in range(len(corr_matrix)):
    for j in range(i,len(corr_matrix)):
        if abs(corr_matrix.iloc[i,j])>=0.2 and corr_matrix.iloc[i,j]!=1.0:
            row_col = i,j
            large_cor[row_col]=corr_matrix.iloc[i,j]
candiadate= np.unique(np.array(list(large_cor.keys())).flatten())

cluster = [[0,2],[1,3,4,5]]   
y_train_train1 = y_train_train[:,np.array(cluster[0])]
y_train_train2 = y_train_train[:,np.array(cluster[1])]
trans_result1 = lp_svm.transform(y_train_train1)
trans_result2 = lp_svm.transform(y_train_train2)
trans_result = [trans_result1]
trans_result.append(trans_result2)

# Fit using lp
lsp_fit = lsp_svm.fit(X_train_train,trans_result,cluster)
lsp_pred = lsp_svm.predict(lsp_fit ,X_train_test,trans_result).toarray()

y_train_test = y_train_test.astype(np.int64)
column_index = np.array([0,2,1,3,4,5])
y_train_test = y_train_test[:,column_index]

print('The accuracy score of music data is ', accuracy_score(lsp_pred,y_train_test))
print('The hamming loss of music data is ', 1-hamming_loss(lsp_pred,y_train_test))
 


##——————————————————run the algorithm on toxic data————————————————————————
## import toxic data
X_train = np.genfromtxt('x_10000sample_2000.csv', delimiter = ',')
y_train = np.genfromtxt('y_10000sample_2000.csv', delimiter = ',')
X_train_train = X_train[0:8000, :]
X_train_test = X_train[8000:, :]
y_train_train = y_train[0:8000, :]
y_train_test = y_train[8000:, :]

y_trainframe = pd.DataFrame(y_train)
corr_matrix = y_trainframe.corr()
large_cor={}
for i in range(len(corr_matrix)):
    for j in range(i,len(corr_matrix)):
        if abs(corr_matrix.iloc[i,j])>=0.2 and corr_matrix.iloc[i,j]!=1.0:
            row_col = i,j
            large_cor[row_col]=corr_matrix.iloc[i,j]
candiadate= np.unique(np.array(list(large_cor.keys())).flatten())

cluster = [[0,1,2,4],[3,5]]
y_train_train1 = y_train_train[:,np.array(cluster[0])]
y_train_train2 = y_train_train[:,np.array(cluster[1])]
trans_result1 = lp_logistic.transform(y_train_train1)
trans_result2 = lp_logistic.transform(y_train_train2)
trans_result = [trans_result1]
trans_result.append(trans_result2)

# Fit using lp
lsp_fit = lsp_logistic.fit(X_train_train,trans_result,cluster)
lsp_pred = lsp_logistic.predict(lsp_fit ,X_train_test,trans_result).toarray()

y_train_test = y_train_test.astype(np.int64)
column_index = np.array([0,2,4,1,3,5])
y_train_test = y_train_test[:,column_index]

print('The accuracy score of toxic data is ', accuracy_score(lsp_pred,y_train_test))
print('The hamming loss of toxic data is ',1-hamming_loss(lsp_pred,y_train_test))
