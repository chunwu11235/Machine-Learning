# -*- coding: utf-8 -*-
"""
Binary Classifiers

Created on Mon Oct 16 14:34:09 2017

@author: ming-chun
"""
import numpy as np


class Binary_Classifier(object):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
        self.p = self.Data.shape[1]
        self.n = self.Data.shape[0]
        
      
    def getData(self):
        return self.Data
        
    def getLabel(self):
        return self.Label

    def train(self):
        raise NotImplementedError('Please implement the training algorithm!')
        
    def predict(self, Data):
        raise NotImplementedError('Please implement the prediction algorithm!')
             
    def predicting_error(self, Data, Label):
        # return testing error
        print('--Testing/Predicting--')
        error = sum( self.predict(Data) != Label)/float(len(Label))
        print('Test Error:',error,'\n')
        return error

class logistic(Binary_Classifier):
    '''
    P(Y=1|X=x) = 1/(1+exp(-theta'x)) where theta is estimated from training data
    '''
    def __init__(self, Data,Label):
        self.n = Data.shape[0]
        self.Data = Data
        self.X = np.c_[np.ones(self.n), Data]
        self.Label = Label
        self.p = self.X.shape[1]
        self.theta = np.zeros( self.p )
    
    def h(X,y,theta):
        return 1.0/(1 + np.exp( - np.diag(y).dot( X ).dot(theta) ) )
         
    def grad(X,y,theta):
        return np.sum( X.T.dot( np.diag( (1-logistic.h(X,y,theta))*y ) )  ,1 )/(-X.shape[0])
    
    def train(self):
        raise NotImplementedError('Please implement the training algorithm!')

    def getClassifier(self):
        return self.theta
    
    def setClassifier(self, theta):
        self.theta = theta

#    def predict_soft(self, X_test):
#        # return Pr(Y=1|X)
#        X_test = np.c_[np.ones(X_test.shape[0]), X_test]
#        return 1.0/( 1+np.exp(-X_test.dot(self.theta)) )

    def predict(self, X_test):
        X = np.c_[np.ones(X_test.shape[0]), X_test]
        return 2*(X.dot(self.theta)>=0)-1
#        return  -1*np.asarray([-1])**(self.predict_soft(X_test)-0.5 >=0)
        
    
class logistic_Newton(logistic):
    def train(self, n_iter = 1000):
        for i in range(n_iter):
            h = logistic.h(self.X,self.Label,self.theta)
            H = (self.X.T).dot( np.diag( h*(1-h) ) ).dot(self.X)/self.n
            grad = logistic.grad(self.X,self.Label,self.theta)
            self.theta -= np.linalg.inv( H ).dot( grad ) 

class  logistic_GD(logistic):
    def train(self, n_iter = 1000, eta = 0.1):
        # eta is the step size used in gradient descent
        for i in range(n_iter):
            self.theta -= eta*logistic.grad(self.X,self.Label,self.theta)

class logistic_SGD(logistic):
    def train(self, batch_size=10 , n_iter = 1000, eta = 0.1):
        n_batch = np.ceil(self.n/float(batch_size))
        index_batch = np.mod(range(self.n),n_batch)
        for i in range(n_iter):
            i_batch = np.mod(i,n_batch)
            X = self.X[ index_batch==i_batch,:]
            y = self.Label[index_batch==i_batch]
            self.theta -= eta*logistic.grad(X,y,self.theta)





class dstump(Binary_Classifier):
    '''
    Decision Stump
    say x = [x1 x2 ... xd ... xp]
    decision stump predicts using y=s*1(xd>=theta) where s = 1 or -1
    s,d,theta are estimated with training data
    '''
     
    def train(self, weight = None, random_search = False):
        '''
        if random_search is True, randomly pick up d among the 1,2,...p
        , otherwise, thoroughly search all possible values
        '''
        if weight is None:
            weight = np.ones(self.n)
           
        parameters = None
        train_error = 1.0
        if random_search:
            d = np.random.randint(self.p)
            xd = self.Data[:,d]
            order = xd.argsort()
            xd = xd[order]
            y = self.Label[order]
            w = weight[order]
            for i in range(self.n):
                if i == 0:
                    theta = xd[i]
                else:
                    theta = (xd[i]+xd[i-1])/float(2)
                for s in [-1,1]:
                    y_hat = s*np.sign( xd-theta )
                    index = y_hat != y
                    error = sum(w[index])/float(sum(w))
                    if error<=train_error:
                        train_error = error
                        parameters = [s,d,theta]

            
        else:
            for d in range(self.p):
                xd = self.Data[:,d]
                order = xd.argsort()
                xd = xd[order]
                y = self.Label[order]
                w = weight[order]
                for i in range(self.n):
                    if i == 0:
                        theta = xd[i]
                    else:
                            theta = (xd[i]+xd[i-1])/float(2)
                    for s in [-1,1]:
                        y_hat = s*np.sign( xd-theta )
                        index = y_hat != y
                        error = sum(w[index])/float(sum(w))
                        if error<=train_error:
                            train_error = error
                            parameters = [s,d,theta]

                
            
        self.parameters = parameters
        self.train_error = train_error
                    


    def getClassifier(self):
        return self.parameters
    
    def setClassifier(self,parameters):
        self.parameters = parameters
    
    def getTrainError(self):
        return self.train_error
    
    def predict(self,Data):
        s,d,theta = self.parameters
        return -s*np.asarray(-1)**(Data[:,d]-theta >= 0) 


## test of dstump
#s,d,theta = 1,0,0.65
#X = np.random.uniform(0,1,(100,3))
#y = 2*(s*X[:,d]>=s*theta)-1
#cls = dstump(X,y)
#cls.train(np.ones(100))
#print('TrainError:',cls.getTrainError())
#print('True Model:',[s,d,theta])
#print('Classifier:',cls.getClassifier())
        

class Adaboost(Binary_Classifier):
    
    def train(self, base_classifier, n_iter = 15 ):
        self.base_classifier = base_classifier
        self.weight = np.ones(self.n)/float(self.n)
        
        self.G = []
        for iteration in range(n_iter):
            cls = base_classifier(self.Data, self.Label)
            cls.train(weight = self.weight)
            error = cls.getTrainError()


            alpha = 0.5*np.log( (1-error)/float(error))
            self.weight *= np.exp(-alpha*(self.Label*cls.predict(self.Data)))
            self.G.append( [ alpha ,cls.getClassifier() ] )

      
    def predict(self,Data):
        cls = self.base_classifier(self.Data,self.Label)
        Gt = np.zeros(Data.shape[0])
        for t in range(len(self.G)):
            alpha, para = self.G[t]            
            cls.setClassifier(para)
            Gt += alpha*cls.predict(Data)            
        return -np.asarray(-1)**(Gt>=0)


       
    def predicting_error(self, Data, Label):
        print('--Testing/Predicting--')
        Error = []
        cls = self.base_classifier(self.Data,self.Label)
        Gt = np.zeros(Data.shape[0])
        for t in range(len(self.G)):
            alpha, para = self.G[t]            
            cls.setClassifier(para)
            Gt += alpha*cls.predict(Data)
            predict = -np.asarray(-1)**(Gt>=0)
            error = np.mean( predict != Label)
            Error.append( error )
        return Error







class decision_tree(Binary_Classifier):
    '''
    binary classification tree ( using Gini index by default)
    '''
    def __init__(self, Data, Label):
        Binary_Classifier.__init__(self, Data,Label)
        self.branch = None # branching rule
        self.left = None
        self.right = None
        self.decision = None # prediction of for this branch
        self.nb = 0 # number of branch
        
        
    def set_nb(self,nb):
        self.nb = nb
        
        
    def Gini(y):
        N = float(len(y))
        if N == 0:
            return 0
        else:
            a = sum( y==1 )/N
            b = sum( y==-1 )/N
        return 1 - (a**2 + b**2)
        
        
    def dstump(X,y,random_search = False, impurity = None):
        if impurity is None:
            impurity = decision_tree.Gini
        
        opt = []
        N = len(y)
        J_min = N
        if random_search:
            d = np.random.randint(X.shape[1])
            xd = X[:,d]
            order = xd.argsort()
            xd,yd = xd[order], y[order]
         
            for n in range(N):
                J = len(yd[n:])*impurity(yd[n:]) + len(yd[:n])*impurity(yd[:n])
                if J <= J_min:
                    J_min = J
                    if n == 0:
                        theta = xd[n]
                    else:
                        theta = (xd[n-1]+ xd[n])/float(2)
                    opt = [d,theta]
        else:           
            for d in range(X.shape[1]):
                xd = X[:,d]
                order = xd.argsort()
                xd,yd = xd[order], y[order]
         
                for n in range(N):
                    J = len(yd[n:])*impurity(yd[n:]) + len(yd[:n])*impurity(yd[:n])
                    if J <= J_min:
                        J_min = J
                        if n == 0:                            
                            theta = xd[n]
                        else:
                            theta = (xd[n-1]+ xd[n])/float(2)
                        opt = [d,theta]
        return opt
        
 
    def train(self,nb_max, random_search = False, impurity = None):
        # nb_max is the maximum depth of the tree
        if impurity is None:
            impurity = decision_tree.Gini
        
        X = self.Data
        y = self.Label
        
        if impurity(y)==0 or self.nb>=nb_max or (X==X[0,:]).all():
            if sum(y==1) >= sum(y==-1):
                self.decision = 1
            else:
                self.decision = -1
        else:
            # branching
            self.branch = decision_tree.dstump(X,y,random_search,impurity)
            d,theta = self.branch
            
            select = ( X[:,d] >= theta ) 

            X_left = X[~select,]
            y_left = y[~select,]
            
            X_right = X[select,]
            y_right = y[select,]            

            
            self.left = decision_tree(X_left,y_left)
            self.left.set_nb(self.nb+1)
            self.left.train(nb_max,random_search,impurity)
            
            self.right = decision_tree(X_right,y_right)
            self.right.set_nb(self.nb+1)
            self.right.train(nb_max,random_search,impurity)
    

            
    def predict(self,X):
        y = []
        for n in range(X.shape[0]):
            xn = X[n,:]
            node = self
            while node.branch is not None:
                # branching
                d, theta = node.branch
                if xn[d] >= theta: # go to the right node
                    node = node.right
                else: # go to the left node
                    node = node.left
            
            y.append(node.decision)
        return np.asarray(y)
                                



              
class random_forest(Binary_Classifier):
    
    def __init__(self, Data,Label):
        self.Data = Data
        self.Label = Label
    
    def train(self, tree_depth, n_trees):
        self.forest = []
        for t in range(n_trees):
            # bootstrap
            sample = np.random.randint(0,len(self.Label),size = len(self.Label))
            X_sample = self.Data[sample,]
            y_sample = self.Label[sample,]
            # create a tree
            tree = decision_tree(X_sample,y_sample)
            tree.train(tree_depth,random_search = True)
            self.forest.append( tree )
            
    
    def predict(self, X):
        G = np.zeros(X.shape[0])
        for tree in self.forest:
            G += tree.predict(X)        
        return -np.asarray(-1)**(G>=0)
    
    def predicting_error(self,X,y):
        print('--Testing/Predicting--')
        Error = []
        G = np.zeros(X.shape[0])
        for tree in self.forest:
            G += tree.predict(X)
            predict = -np.asarray(-1)**(G>=0)
            error = np.mean(predict != y)
            Error.append(error)
        return Error







