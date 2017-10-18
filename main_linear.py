#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Play with linear classifiers

Created on Tue Oct 17 16:29:35 2017

@author: ming-chun
"""
import numpy as np
import matplotlib.pyplot as plt
from binary_classifier import*
from kernel_LDA import*


def plot_decision_boundary(classifier):
    X = classifier.getData()
    y = classifier.getLabel()
    pred_func = classifier.predict
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() , X[:, 0].max() 
    y_min, y_max = X[:, 1].min() , X[:, 1].max()
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
    plt.xlim([x_min,x_max])
    plt.ylim([y_min,y_max])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)





# generate data
n_train = 200
X_train  = np.random.rand(n_train,2)
y_train = 2*(X_train[:,0]<=0.3)-1
noise = -2*np.random.binomial(1,0.1,n_train)+1
y_train = noise*y_train

n_test = 100
X_test  = np.random.rand(n_test,2)
y_test = 2*(X_test[:,0]<=0.3)-1
noise = -2*np.random.binomial(1,0.1,n_test)+1
y_test = noise*y_test



# logistic Newton
print('Logistic Newton')
cls_newton = logistic_Newton(X_train,y_train)
cls_newton.train(n_iter = 1000)
cls_newton.predicting_error(X_test,y_test)

plt.figure()
plot_decision_boundary(cls_newton)
plt.title('Logistic Newton - Training')


# logistic GD
print('Logistic Gradient Descent')
n_iter = 1000
step_size = 0.1
cls_gd = logistic_GD(X_train,y_train)
cls_gd.train(n_iter,step_size)
cls_gd.predicting_error(X_test,y_test)

plt.figure()
plot_decision_boundary(cls_gd)
plt.title('Logistic Gradient Descent - Training')



# logistic SGD
print('Logistic Stochastic Gradient Descent')
n_iter = 1000
step_size = 0.1
batch_size = 10

cls_sgd = logistic_SGD(X_train,y_train)
cls_sgd.train(batch_size,n_iter,step_size)
cls_sgd.predicting_error(X_test,y_test)

plt.figure()
plot_decision_boundary(cls_sgd)
plt.title('Logistic Stochastic Gradient Descent - Training')

# LDA
print('LDA')
cls_lda = LDA(X_train, (y_train+1)/2 )
cls_lda.train()
cls_lda.predicting_error(X_test, (y_test+1)/2)

plt.figure()
plot_decision_boundary(cls_lda)
plt.title('LDA - Training')


