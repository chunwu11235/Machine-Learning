# -*- coding: utf-8 -*-
"""
Play with nonlinear classifiers


Created on Tue Oct 17 16:41:28 2017

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
    h = 0.02
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
y_train = 2*(np.sum(X_train**2,axis = 1)<=0.5)-1
noise = -2*np.random.binomial(1,0.1,n_train)+1
y_train = noise*y_train

n_test = 100
X_test  = np.random.rand(n_test,2)
y_test = 2*(np.sum(X_test**2,axis = 1)<=0.5)-1
noise = -2*np.random.binomial(1,0.1,n_test)+1
y_test = noise*y_test




# Kernel Smoothing LDA
print('Kenel Smoothing LDA')
k = Gaussian(0.5)
print(k)
cls_klda= Local_LDA(X_train, (y_train+1)/2)
cls_klda.setKernel(k)
cls_klda.predicting_error(X_test, (y_test+1)/2)

plt.figure()
plot_decision_boundary(cls_klda)
plt.title('Kernel LDA')
plt.show()


# Decision Tree
print('Decision Tree')
tree_depth = 5
cls_dt = decision_tree(X_train,y_train)
cls_dt.train(tree_depth)
cls_dt.predicting_error(X_test,y_test)

plt.figure()
plot_decision_boundary(cls_dt)
plt.title('Decision Tree')



# Adaboost
cls_ada = Adaboost(X_train,y_train)
cls_ada.train(dstump,300)
ada_train = cls_ada.predicting_error(X_train,y_train)
ada_test = cls_ada.predicting_error(X_test,y_test)

plt.figure()
plt.plot(ada_train,label = 'training')
plt.plot(ada_test,label = 'testing')
plt.xlabel('number of decision stumps')
plt.ylabel('error rate')
plt.title('Adaboost wirh Decision Stump')
plt.legend()
plt.show()

plt.figure()
plot_decision_boundary(cls_ada)
plt.title('Adaboost')


# Random Forest
tree_dept = 2
n_tree = 300
cls_rf = random_forest(X_train,y_train)
cls_rf.train(tree_dept,n_tree)
rf_train = cls_rf.predicting_error(X_train,y_train)
rf_test = cls_rf.predicting_error(X_test,y_test)


plt.figure()
plt.plot(rf_train,label = 'training')
plt.plot(rf_test,label = 'testing')
plt.xlabel('number of decision stumps')
plt.ylabel('error rate')
plt.title('Random Forest')
plt.legend()
plt.show()

plt.figure()
plot_decision_boundary(cls_rf)
plt.title('Random Forest')


