# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:37:29 2017

Zipcode

@author: ming-chun
"""

import numpy as np
from kernel_LDA import*

# ZIPCODE Data

# load training data
Data_train = np.loadtxt(fname = 'zip_train.txt')
Label_train = np.asarray( Data_train[:,0], dtype = int)
Data_train = Data_train[:,1:]


# load test data
Data_test = np.loadtxt(fname = 'zip_test.txt')
Label_test = np.asarray( Data_test[:,0], dtype = int)
Data_test = Data_test[:,1:]

#LDA
print('LDA Zipcode')
model_LDA = LDA(Data_train, Label_train)
model_LDA.train()
LDA_error = model_LDA.predicting_error(Data_test, Label_test)


# Kernel Smoothing LDA
print('Kenel LDA Zipcode')
G = Gaussian(3)
print(G)
model_LLDA= Local_LDA(Data_train, Label_train)
model_LLDA.setKernel(G)
LLDA_error = model_LLDA.predicting_error(Data_test, Label_test,verbose=True)

