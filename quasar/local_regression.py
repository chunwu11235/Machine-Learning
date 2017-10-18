# -*- coding: utf-8 -*-
"""
Local Regression for smoothing quasar data
For more details, please see Stanford CS229 2016 problem 1.5

@author: Ming-Chun Wu
"""

import numpy as np
import matplotlib.pyplot as plt

# load data
D = np.loadtxt('quasar_train.csv', delimiter=",")
X = np.c_[ np.ones( D.shape[1]) , D[0,:].T ]
Y = D[1:,:]
del D

y = Y[0,].T


# Linear Regression
theta_lin = np.linalg.inv( X.T.dot(X) ).dot( X.T ).dot(y)


# local regression
def local_regression(X,y, x_test, tau):
    w = np.exp(  ((X[:,1]-x_test)**2)/float(-2*(tau**2) ) )
    W = np.diag(w)
    theta_local = np.linalg.inv( X.T.dot(W).dot(X) ).dot( X.T.dot(W) ).dot(y)
    y_predict = np.array([ 1, x_test]).dot(theta_local)
    return y_predict




# visualization of the first data sample
plt.plot(X[:,1],y,'kx', label = 'raw data')
plt.plot(X[:,1],X.dot(theta_lin),'r--',linewidth= 2, label = 'linear regression' )
plt.plot(X[:,1],[ local_regression(X,y, x_test, 1) for x_test in X[:,1] ],linewidth= 1, label = 'Bandwidth 1')
plt.plot(X[:,1],[ local_regression(X,y, x_test, 10) for x_test in X[:,1] ],linewidth= 1, label = 'Bandwidth 10' )
plt.plot(X[:,1],[ local_regression(X,y, x_test, 100) for x_test in X[:,1] ],linewidth= 1,label = 'Bandwidth 100' )
plt.plot(X[:,1],[ local_regression(X,y, x_test, 1000) for x_test in X[:,1] ],linewidth= 1, label = 'Bandwidth 1000' )
plt.legend()
plt.xlabel('wavelength')
plt.ylabel('Flux')
plt.title('Gaussian Local Regression for Smoothing Quasar Data')
plt.savefig('quasar_smoothing.pdf')





# smoothing data
Y_train_smooth = Y
for m in range( Y.shape[0] ):
    y = Y[m,].T
    Y_train_smooth[m,:] =  np.asarray( [ local_regression(X, y, x, 5)  for x in X[:,1]   ]  ).T
del Y
np.savetxt('quasar_train_smooth.txt',Y_train_smooth)


D = np.loadtxt('quasar_test.csv', delimiter=",")
X = np.c_[ np.ones( D.shape[1]) , D[0,:].T ]
Y = D[1:,:]

Y_test_smooth = Y
for m in range( Y.shape[0] ):
    y = Y[m,].T
    Y_train_smooth[m,:] =  np.asarray( [ local_regression(X, y, x, 5)  for x in X[:,1]   ]  ).T
del Y
np.savetxt('quasar_test_smooth.txt',Y_train_smooth)