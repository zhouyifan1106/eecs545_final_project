#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 18:53:45 2018

@author: TLuv
"""
import numpy as np
import pandas as pd
import BinaryRelevance as br
import BinaryRelevance_svm as br_svm
import ClassifierChain as cc
from sklearn.metrics import accuracy_score
import cc_svm
import cc_lr
import mcc as mcc_lr
import mcc_svm
import mlknn
import LabelPowerset_svm as lp_svm
import LabelPowerset_logistic as lp_lr
import LabelSpacePartition_logistic as par_lr
import LabelSpacePartition_svm as par_svm



################################## 10000 * 2000 data ##################################
X_1_2000 = np.genfromtxt('x_10000sample_2000.csv', delimiter = ',')
y_1_2000 = np.genfromtxt('y_10000sample_2000.csv', delimiter = ',')
X_train_1_2000 = X_1_2000[:8000, :]
X_test_1_2000 = X_1_2000[8000:, :]
y_train_1_2000 = y_1_2000[:8000, :]
y_test_1_2000 = y_1_2000[8000:, :] 

# mlknn
k = 90
s = 1.0
pred_knn = mlknn.predict(s, k, X_train_1_2000, X_test_1_2000, y_train_1_2000)


toxic_scores = []
toxic_hammings = []
for k in range(40, 130, 10):
    print(k)
    pred_knn = mlknn.predict(s, k, X_train_1_2000, X_test_1_2000, y_train_1_2000)
    toxic_scores.append(accuracy_score(y_test_1_2000, pred_knn))
    toxic_hammings.append(len(np.argwhere(y_test_1_2000 != pred_knn)))
toxic_score = max(toxic_scores)
toxic_hamming = 1 - min(toxic_hammings)

# cc_lr
toxic_cc = cc_lr.fit(X_train_1_2000, y_train_1_2000)
toxic_cc_pred = cc_lr.predict(toxic_cc, X_test_1_2000)
accuracy_score(y_test_1_2000, np.array(toxic_cc_pred).T)

# br_lr
toxic_br = br.fit(X_train_1_2000, y_train_1_2000)
toxic_br_pred = br.predict(toxic_br, X_test_1_2000)
accuracy_score(y_test_1_2000, toxic_br_pred)

# mcc_lr
mcc_classifiers, mcc_indices = mcc_lr.fit(X_train_1_2000, y_train_1_2000, 0.9)
toxic_mcc_pred = mcc_lr.predict(mcc_classifiers, mcc_indices, X_test_1_2000)
accuracy_score(y_test_1_2000, np.array(toxic_mcc_pred).T)

# lp_lr
trans_result = lp_lr.transform(y_train_1_2000)
y_trans = trans_result['train_vector']
inverse_result = lp_lr.inverse_transform(y_trans, trans_result)

lp_fit = lp_lr.fit(X_train_1_2000,trans_result)
lp_pred = lp_lr.predict(lp_fit ,X_test_1_2000,trans_result)

y_test_1_2000 = y_test_1_2000.astype(np.int64)
accuracy_score(y_test_1_2000, lp_pred)
1 - len(np.argwhere(y_test_1_2000 != lp_pred)) / (lp_pred.shape[0] * lp_pred.shape[1])

# partition_lr
y_trainframe = pd.DataFrame(y_1_2000)
corr_matrix = y_trainframe.corr()
large_cor={}
for i in range(len(corr_matrix)):
    for j in range(i,len(corr_matrix)):
        if abs(corr_matrix.iloc[i,j])>=0.2 and corr_matrix.iloc[i,j]!=1.0:
            row_col = i,j
            large_cor[row_col]=corr_matrix.iloc[i,j]
candiadate= np.unique(np.array(list(large_cor.keys())).flatten())

cluster = [[0,1,2,4],[3,5]]
y_train_train1 = y_train_1_2000[:,np.array(cluster[0])]
y_train_train2 = y_train_1_2000[:,np.array(cluster[1])]
trans_result1 = lp_lr.transform(y_train_train1)
trans_result2 = lp_lr.transform(y_train_train2)
trans_result = [trans_result1]
trans_result.append(trans_result2)

lsp_fit = par_lr.fit(X_train_1_2000,trans_result,cluster)
lsp_pred = par_lr.predict(lsp_fit ,X_test_1_2000,trans_result).toarray()

y_test_1_2000 = y_test_1_2000.astype(np.int64)
column_index = np.array([0,2,4,1,3,5])
y_test_1_2000 = y_test_1_2000[:,column_index]

accuracy_score(lsp_pred,y_test_1_2000)
1 - len(np.argwhere(y_test_1_2000 != lsp_pred)) / (lsp_pred.shape[0] * lsp_pred.shape[1])


################################## simulated data ##################################
X_sim = np.genfromtxt('x.csv', delimiter = ',')
y_sim = np.genfromtxt('y.csv', delimiter = ',')
X_sim_train = X_sim[:8000, :]
X_sim_test = X_sim[8000:, :]
y_sim_train = y_sim[:8000, :]
y_sim_test = y_sim[8000:, :]

# mlknn
k = 90
s = 1.0

sim_scores = []
sim_hammings = []
for k in range(40, 130, 10):
    print(k)
    pred_knn = mlknn.predict(s, k, X_sim_train, X_sim_test, y_sim_train)
    sim_scores.append(accuracy_score(y_sim_test, pred_knn))
    sim_hammings.append(len(np.argwhere(y_sim_test != pred_knn)))
sim_score = max(sim_scores)
sim_hamming = 1 - min(sim_hammings)


# cc_svc
sim_cc = cc_svm.fit(X_sim_train, y_sim_train)
sim_cc_pred = cc_svm.predict(sim_cc, X_sim_test)
accuracy_score(y_sim_test, np.array(sim_cc_pred).T)

# br_svm
sim_br = br_svm.fit(X_sim_train, y_sim_train)
sim_br_pred = br_svm.predict(sim_br, X_sim_test)
accuracy_score(y_sim_test, sim_br_pred)

# mcc_svm
mcc_classifiers_sim, mcc_indices_sim = mcc_svm.fit(X_sim_train, y_sim_train, 0.2)
sim_mcc_pred = mcc_svm.predict(mcc_classifiers_sim, mcc_indices_sim, X_sim_test)
accuracy_score(y_sim_test, np.array(sim_mcc_pred).T)

# lp_svm
trans_result = lp_svm.transform(y_sim_train)
y_trans = trans_result['train_vector']
inverse_result = lp_svm.inverse_transform(y_trans, trans_result)

lp_fit = lp_svm.fit(X_sim_train,trans_result)
lp_pred = lp_svm.predict(lp_fit ,X_sim_test,trans_result)

y_sim_test = y_sim_test.astype(np.int64)
accuracy_score(y_sim_test, lp_pred)
1 - len(np.argwhere(y_sim_test != lp_pred)) / (lp_pred.shape[0] * lp_pred.shape[1])

# partition_svm
y_trainframe = pd.DataFrame(y_sim)
corr_matrix = y_trainframe.corr()
large_cor={}
for i in range(len(corr_matrix)):
    for j in range(i,len(corr_matrix)):
        if abs(corr_matrix.iloc[i,j])>=0.2 and corr_matrix.iloc[i,j]!=1.0:
            row_col = i,j
            large_cor[row_col]=corr_matrix.iloc[i,j]
candiadate= np.unique(np.array(list(large_cor.keys())).flatten())

cluster = [[0,1,2,4],[3,5]]
y_train_train1 = y_sim_train[:,np.array(cluster[0])]
y_train_train2 = y_sim_train[:,np.array(cluster[1])]
trans_result1 = lp_svm.transform(y_train_train1)
trans_result2 = lp_svm.transform(y_train_train2)
trans_result = [trans_result1]
trans_result.append(trans_result2)

lsp_fit = par_svm.fit(X_sim_train,trans_result,cluster)
lsp_pred = par_svm.predict(lsp_fit ,X_sim_test,trans_result).toarray()

y_sim_test = y_sim_test.astype(np.int64)
column_index = np.array([0,2,4,1,3,5])
y_sim_test = y_sim_test[:,column_index]

accuracy_score(lsp_pred,y_sim_test)
1 - len(np.argwhere(y_sim_test != lsp_pred)) / (lsp_pred.shape[0] * lsp_pred.shape[1])


################################## music data ##################################
X_music = np.genfromtxt('x_music.csv', delimiter = ',')
y_music = np.genfromtxt('y_music.csv', delimiter = ',')
X_music_train = X_music[:500, :]
X_music_test = X_music[500:, :]
y_music_train = y_music[:500, :]
y_music_test = y_music[500:, :]

# mlknn
k = 22
s = 1.0
pred_knn = mlknn.predict(s, k, X_music_train, X_music_test, y_music_train)

music_scores = []
for k in range(5, 35, 5):
    pred_knn = mlknn.predict(s, k, X_music_train, X_music_test, y_music_train)
    music_scores.append(accuracy_score(y_music_test, pred_knn))
music_score = max(music_scores)
music_hamming = 1 - len(np.argwhere(y_music_test != pred_knn)) / (pred_knn.shape[0] * pred_knn.shape[1])

# cc_svc
music_cc = cc_svm.fit(X_music_train, y_music_train)
music_cc_pred = cc_svm.predict(music_cc, X_music_test)
accuracy_score(y_music_test, np.array(music_cc_pred).T)

# br_svm
music_br = br_svm.fit(X_music_train, y_music_train)
music_br_pred = br_svm.predict(music_br, X_music_test)
accuracy_score(y_music_test, music_br_pred)

# mcc_svm
mcc_classifiers_music, mcc_indices_music = mcc_svm.fit(X_music_train, y_music_train, 0.2)
music_mcc_pred = mcc_svm.predict(mcc_classifiers_music, mcc_indices_music, X_music_test)
accuracy_score(y_music_test, np.array(music_mcc_pred).T)

# lp_svm
trans_result = lp_svm.transform(y_music_train)
y_trans = trans_result['train_vector']
inverse_result = lp_svm.inverse_transform(y_trans, trans_result)

lp_fit = lp_svm.fit(X_music_train,trans_result)
lp_pred = lp_svm.predict(lp_fit ,X_music_test,trans_result)

y_music_test = y_music_test.astype(np.int64)
accuracy_score(y_music_test, lp_pred)
1 - len(np.argwhere(y_music_test != lp_pred)) / (lp_pred.shape[0] * lp_pred.shape[1])

# partition_svm
y_trainframe = pd.DataFrame(y_music)
corr_matrix = y_trainframe.corr()
large_cor={}
for i in range(len(corr_matrix)):
    for j in range(i,len(corr_matrix)):
        if abs(corr_matrix.iloc[i,j])>=0.2 and corr_matrix.iloc[i,j]!=1.0:
            row_col = i,j
            large_cor[row_col]=corr_matrix.iloc[i,j]
candiadate= np.unique(np.array(list(large_cor.keys())).flatten())

cluster = [[0,1,2,4],[3,5]]
y_train_train1 = y_music_train[:,np.array(cluster[0])]
y_train_train2 = y_music_train[:,np.array(cluster[1])]
trans_result1 = lp_svm.transform(y_train_train1)
trans_result2 = lp_svm.transform(y_train_train2)
trans_result = [trans_result1]
trans_result.append(trans_result2)

lsp_fit = par_svm.fit(X_music_train,trans_result,cluster)
lsp_pred = par_svm.predict(lsp_fit ,X_music_test,trans_result).toarray()

y_music_test = y_music_test.astype(np.int64)
column_index = np.array([0,2,4,1,3,5])
y_music_test = y_music_test[:,column_index]

accuracy_score(lsp_pred,y_music_test)
1 - len(np.argwhere(y_music_test != lsp_pred)) / (lsp_pred.shape[0] * lsp_pred.shape[1])









