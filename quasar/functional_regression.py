# -*- coding: utf-8 -*-
"""
Quasar Signal Prediction using Function Regression 

@auther: Ming-Chun Wu
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt('quasar_train.csv', delimiter=",")[0,:]
X = np.c_[ np.ones( X.shape ), X]
F_train = np.loadtxt('quasar_train_smooth.txt')
F_test = np.loadtxt('quasar_test_smooth.txt')


# Functional Regression

def distance(f1,f2):
    return np.sum( (f1-f2)**2 )
    
def neighbor(F,f,k):
    d = np.asarray( [ distance(fi,f) for fi in F ] )#distance
    index = d.argsort()[:k] # index of the k nearest neighbor
    h = d[ index[k-1] ]
    return index,d[index]/float(h)


def nonpara_regression(F, lam, f_right, k):
    right = lam>=1300
    left = lam<1200
    index, d_h = neighbor(F[:,right], f_right, k)
    k = np.maximum( 1-d_h,0)    
    f_left = np.sum( np.diag(k).dot( F[index][:,left] )/float(sum(k)),0   )
    return f_left
    



# training
F_train_left = np.asarray( [nonpara_regression(F_train,X[:,1], f,3) for f in F_train[ :,X[:,1]>=1300 ] ] )
error_train = [ distance( F_train_left[m,:], F_train[m,X[:,1]<1200  ]  )  for m in range(F_train.shape[0]) ]


# testing
F_test_left = np.asarray( [nonpara_regression(F_train,X[:,1], f,3) for f in F_test[ :,X[:,1]>=1300 ] ] )
error_test = [ distance( F_test_left[m,:], F_test[m,X[:,1]<1200  ]  )  for m in range(F_test.shape[0]) ]



# visualization
plt.figure()
plt.plot(F_test[0,:], label = 'true signal')
plt.plot(F_test_left[0,:],'r', label = 'prediction')
plt.legend()
plt.xlabel('wavelength')
plt.ylabel('Flux')
plt.title('Predict Quasar Signal Using Functional Regression\n(1-th Testing Sample)')


plt.figure()
plt.plot(F_test[5,:], label = 'true signal')
plt.plot(F_test_left[5,:],'r', label = 'prediction')
plt.legend()
plt.xlabel('wavelength')
plt.ylabel('Flux')
plt.title('Predict Quasar Signal Using Functional Regression\n(6-th Testing Sample)')
