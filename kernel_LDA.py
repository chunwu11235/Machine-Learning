# -*- coding: utf-8 -*-
"""
LDA and Kernel Smoothing LDA

@author: Ming-Chun Wu


For more details, please refer to "The Elements of Statistical Learning Chap. 6"
"""

import numpy as np
from numpy.matlib import repmat

class Classifier(object):
    def __init__(self, D_train, L_train):
        self.Data = D_train
        self.Label = L_train
        self.n = D_train.shape[0]
        self.p = D_train.shape[1]
        self.m = len(set(L_train))#max(L_train)+1
      
    def getData(self):
        return self.Data
        
    def getLabel(self):
        return self.Label

    def train(self):
        raise NotImplementedError('Please implement the training algorithm!')
        
    def predict(self, Data):
        raise NotImplementedError('Please implement the prediction algorithm!')
             
    def predicting_error(self, Data, Label):
        print('--Testing/Predicting--')
        error = sum( self.predict(Data) != Label)/float(len(Label))
        print('Test Error:',error,'\n')
        return error


class LDA(Classifier):
    
    def __init__(self, D_train, L_train):
        Classifier.__init__(self, D_train,L_train)
#        self.weight = np.asarray([1.0]*self.n)
        self.weight = np.ones(self.n)

        
        
    def train(self):
        weight = self.weight
        w_total_sum = float(weight.sum())

        Mu = np.zeros([self.m, self.p])
        Pi = np.zeros(self.m)
        Sigma = np.zeros([self.p, self.p])
        
        for i in range(self.m):
            index = self.Label == i
            w_partial_sum =  float(weight[index].sum() )

            Pi[i] = w_partial_sum/w_total_sum
            W = np.diag(weight[index])
            X = self.Data[index,:]
            mu = ( (W/w_partial_sum).dot( X ) ).sum(0)
            Mu[i, :] = mu
            
            X_tilde = X - repmat( mu, X.shape[0],1 )
            Sigma += X_tilde.T.dot(W/w_total_sum).dot( X_tilde  )
        self.Mu = Mu
        self.Pi = Pi
        self.Sigma = Sigma
        self.Sigma_inv = np.linalg.inv(Sigma)

        
        
        
    def score(self, input_data):
        return input_data.dot(self.Sigma_inv).dot(self.Mu.T) \
            - 0.5*np.diag( self.Mu.dot(self.Sigma_inv).dot(self.Mu.T) ) \
            +np.log(self.Pi)
            

    def predict(self, Data):
        pred = []
        try:
            Data.shape[1]
            for x in Data:
                pred.append(np.argmax( self.score(x) ))                    
        except IndexError: # Data has only one sample
            return np.argmax( self.score(Data) )
 
        return np.asarray(pred)



class Local_LDA(LDA):

            
    def setKernel(self, Kernel):
        self.Kernel = Kernel

        
    def compute_weight(self, input_data):
        weight = np.zeros(self.n)
        for i in range(self.n):
            weight[i] = self.Kernel.kernel( input_data, self.Data[i]) 
        self.weight = weight

        
    def train(self,input_data):
        self.compute_weight(input_data)
        LDA.train(self)
        

        
    def predict(self, Data):          
        
        if len(Data.shape) == 1:
            self.train(Data)
            return LDA.predict(self, Data)
        else:
            pred = []
            for x in Data:
                self.train(x)
                p = LDA.predict(self, x)
                pred.append( p )
            return np.asarray(pred)

        
    def predicting_error(self, Data, Label, verbose = False):
        print('--Testing/Predicting--')
        n = len(Label)
        error = 0.0
        for i in range(n):
            if self.predict(Data[i]) != Label[i]:
                error += 1
                if verbose:
                    print('predict error at ',i,'-th testing example')
        error = error/float(n)
        print('Test Error:',error,'\n')
        return error/float(n)



class Kernel(object):
    def __init__(self, bandwidth = 1):
        self.bandwidth = bandwidth
        self.name = 'Kernel'
    
    def setBandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        
    def kernel(self,x0,x):
        raise NotImplementedError('This Kernel is not implemented.')
        
    def __str__(self):
        return self.name+' kernel with bandwidth '+str(self.bandwidth)
        
class Gaussian(Kernel):
    def __init__(self, b):
        Kernel.__init__(self,b)
        self.name = 'Gaussian'
        
    def kernel(self,x0,x):
            p = float(len(x0))
            b = self.bandwidth
            return (2.0*np.pi*(b**2))**(-p/2.0) * np.exp( -0.5*(np.linalg.norm( x0-x )/b)**2  )
        





            
            
            


    
        
