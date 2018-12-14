#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 19:35:55 2018

@author: naichen
"""

import numpy as np
import LabelPowerset as lp
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

trans_result = lp.transform(y_train_train)
y_trans = trans_result['train_vector']
inverse_result = lp.inverse_transform(y_trans, trans_result)

# Fit using lp
lp_fit = lp.fit(X_train_train,trans_result)
lp_pred = lp.predict(lp_fit ,X_train_test,trans_result)

print('The accuracy score of simulated data is ', accuracy_score(lp_pred,y_train_test))
print('The hamming loss of simulated data is ',1-hamming_loss(lp_pred,y_train_test))


##——————————————————run the algorithm on music data————————————————————————
## import music data

X_train = np.genfromtxt('x_music.csv', delimiter = ',')
y_train = np.genfromtxt('y_music.csv', delimiter = ',')

X_train_train = X_train[0:499, :]
X_train_test = X_train[500:, :]
y_train_train = y_train[0:499, :]
y_train_test = y_train[500:, :]

trans_result = lp.transform(y_train_train)
y_trans = trans_result['train_vector']
inverse_result = lp.inverse_transform(y_trans, trans_result)

# Fit using lp
lp_fit = lp.fit(X_train_train,trans_result)
lp_pred = lp.predict(lp_fit ,X_train_test,trans_result)

print('The accuracy score of music data is ',accuracy_score(lp_pred,y_train_test))
print('The hamming loss of music data is ',1-hamming_loss(lp_pred,y_train_test))

##——————————————————run the algorithm on toxic data————————————————————————
## import toxic data

X_train = np.genfromtxt('x_10000sample_2000.csv', delimiter = ',')
y_train = np.genfromtxt('y_10000sample_2000.csv', delimiter = ',')
X_train_train = X_train[0:8000, :]
X_train_test = X_train[8000:, :]
y_train_train = y_train[0:8000, :]
y_train_test = y_train[8000:, :]

trans_result = lp.transform(y_train_train)
y_trans = trans_result['train_vector']
inverse_result = lp.inverse_transform(y_trans, trans_result)

# Fit using lp
lp_fit = lp.fit(X_train_train,trans_result)
lp_pred = lp.predict(lp_fit ,X_train_test,trans_result)

y_train_test = y_train_test.astype(np.int64)
print('The accuracy score of toxic data is ',accuracy_score(lp_pred,y_train_test))
print('The hamming loss of toxic data is ',1-hamming_loss(lp_pred,y_train_test))
