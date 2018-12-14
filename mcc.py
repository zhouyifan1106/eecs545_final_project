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
        classifier = LogisticRegression()#class_weight = 'balanced')
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


#x_train = genfromtxt('x_train_1000_features.csv', delimiter=',') #21519*1000
#x_test = genfromtxt('x_test_1000_features.csv', delimiter=',') #7077*1000
#y_train = genfromtxt('y_train.csv', delimiter=',')#21519*22
#y_train = y_train.astype(np.int)#7077*22
#y_test = genfromtxt('y_test.csv', delimiter=',')
#y_test = y_test.astype(np.int)
#
#
#classifiers, indices = fit(x_train,y_train,0.9)
#y_pred = predict(classifiers,indices,x_test)
#y_pred = np.array(y_pred).T
#y_pred = y_pred.astype("int")
#accuracy_score(y_test,y_pred)
#
#
#f1_score(y_test,y_pred,average="weighted")









#==============================================================================
# count= 0
# for r in range(0,7077):
#     for c in range(0,22):
#         if y_test[r,c] == y_pred[r,c]:
#             count += 1
# # 141363
# # 90.79540637404139
# # 0.06726013847675569
# 
# 
# # 110231
# # 70.79977391550092
# # 0.0031086618623710613
# 
# from sklearn.metrics import accuracy_score
# 
# count/(22*7077)*100
# accuracy_score(y_test,y_pred)
# 
# 
# 
# classifiers = fit(x_train,y_train)
# y_pred = predict(classifiers,x_test) #
# y_pred = y_pred.toarray()
# # 141347
# # 90.78512980590132
# # 0.06768404691253356
# 
#==============================================================================




