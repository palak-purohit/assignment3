# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
from autograd import grad
import autograd.numpy as np
from matplotlib.patches import FancyArrowPatch
import math
from tqdm import trange
class LogisticRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept 
        self.coef_ = None 
        self.X_new = None
        self.y_new = None
        self.ld = None
        self.theta=None
        self.theta_copy =None
        self.labels=None
        pass


    def cost_func(self,theta): #cost function is mse for all examples
        X = self.X_new 
        y = self.y_new
        X = np.array(X) 
        y = np.array(y)
        y_hat = self.sigmoid(np.dot(X,theta))
        T1 =  -np.dot(y.T,np.log(y_hat))
        T2 = -np.dot((np.ones(y.shape)-y).T,np.log(np.ones(y.shape)-y_hat))
        cross_entropy = np.sum(T1 + T2)
        return cross_entropy

    def cost_func_L2(self,theta): #cost function is mse for all examples
        X = self.X_new 
        y = self.y_new
        X = np.array(X) 
        y = np.array(y)
        y_hat = self.sigmoid(np.dot(X,theta))
        T1 =  -np.dot(y.T,np.log(y_hat))
        T2 = -np.dot((np.ones(y.shape)-y).T,np.log(np.ones(y.shape)-y_hat))
        cross_entropy = np.sum(T1 + T2)
        cost= cross_entropy + self.ld*np.array((theta)*(theta.transpose()))[0]
        return cost

    def cost_func_L1(self,theta): #cost function is mse for all examples
        X = self.X_new 
        y = self.y_new
        X = np.array(X) 
        y = np.array(y)
        y_hat = self.sigmoid(np.dot(X,theta))
        T1 =  -np.dot(y.T,np.log(y_hat))
        T2 = -np.dot((np.ones(y.shape)-y).T,np.log(np.ones(y.shape)-y_hat))
        cross_entropy = np.sum(T1 + T2)
        cost= cross_entropy + self.ld*np.sum(np.abs(theta))
        return cost

    def cross_entropy_multi(self, theta):
        
        P = np.exp(np.dot(np.array(self.X_new),theta)) 
        P /= np.sum(P,axis=1).reshape(-1,1)
        cross_entropy = 0
        for k in range(len(self.labels)):
            cross_entropy -= np.dot(np.array(self.y_new == k,dtype =float),np.log(P[:,k]))  
                            
        return np.array(cross_entropy)

    def sigmoid(self,z):
        return 1./(1+np.exp(-np.array(z,dtype=float)))

    def softmax(self,X,cl,theta) : 
        
        den =0
        out = np.exp(np.dot(X,theta))
        return out[:,cl] / np.sum(out,axis=1)

    def maximum_coef(self,X,y): # returns coefficient with maximum magnitude
        max_theta = abs(np.amax(self.coef_))
        return max_theta

    def fit_vectorised(self, X, y,batch_size=5, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        rows = len(X)
        no_batch = rows//batch_size # total number of batches

        learn_rate = lr
        if(self.fit_intercept): # add column of 1s in the beginning if fit intercept is true
            ls = pd.Series(np.ones(len(X)))
            X = pd.concat([ls,X],axis=1, ignore_index=True)
        df = pd.concat([X,y.rename("y")],axis=1, ignore_index=True)
        df_split = [df.iloc[i*batch_size:(i+1)*batch_size] for i in range(no_batch)] #divide entire dataframe into batches

        theta = np.zeros(len(X.columns)).flatten() #initializing theta with zeros
        for i in range(n_iter):
            if lr_type=='inverse':# if type is inverse, lr becomes original lr / iteration number
                lr = learn_rate/(i+1)
            df2 = df_split[i%no_batch] #choose batch (X,y) for this iteration
            df2 = df2.reset_index(drop=True)
            X_iter = df2.iloc[:,0:len(df2.columns)-1]
            y_iter = df2.iloc[:,-1]
            xdash = X_iter.transpose()
            y_pred = np.dot(X_iter, theta) #calculating y_pred for entire matrix
            loss = self.sigmoid(y_pred)-y_iter
            gradient = xdash.dot(loss)/len(y_pred) 
            theta = theta - lr * gradient #updating theta once in every iteration
            # storing theta0 and theta1 after each iteration      
        self.coef_= theta
        pass





    def fit_L1(self, X, y, batch_size=1, n_iter=100, lr=0.01, ld=1,lr_type='constant'):

        rows = len(X)
        no_batch = rows//batch_size # total number of batches
        self.ld=ld
        learn_rate = lr
        if(self.fit_intercept): # add column of 1s in the beginning if fit intercept is true           
            ls = pd.Series(np.ones(len(X)))
            X = pd.concat([ls,X],axis=1, ignore_index=True)
        df = pd.concat([X,y.rename("y")],axis=1, ignore_index=True)
        df_split = [df.iloc[i*batch_size:(i+1)*batch_size] for i in range(no_batch)] #divide entire dataframe into batches

        theta =  np.random.randn(len(X.columns)).flatten() #initializing theta with zeros
        for i in range(n_iter):
            if lr_type=='inverse': # if type is inverse, lr becomes original lr / iteration number
                lr = learn_rate/(i+1)
            df2 = df_split[i%no_batch] #choose batch (X,y) for this iteration
            df2 = df2.reset_index(drop=True)
            X_iter = df2.iloc[:,0:len(df2.columns)-1]
            y_iter = df2.iloc[:,-1]
            self.X_new = X_iter
            self.y_new = y_iter
            xdash = X_iter.transpose()
            y_pred = np.dot(X_iter, theta) #calculating y_pred for entire matrix
            # loss = sigmoid(y_pred)-y_iter
            thet = np.array(theta) 
            gradient = grad(self.cost_func_L1)
            
            gradient_fin = gradient(thet)
            gradient_fin =np.array([0 if math.isnan(i) else i for i in gradient_fin ])
            theta = theta - lr * gradient_fin/batch_size #updating theta            
        self.coef_= theta

        pass

    def fit_L2(self, X, y, batch_size=1, n_iter=100, lr=0.01, ld=1,lr_type='constant'):

        rows = len(X)
        no_batch = rows//batch_size # total number of batches
        self.ld=ld
        learn_rate = lr
        if(self.fit_intercept): # add column of 1s in the beginning if fit intercept is true           
            ls = pd.Series(np.ones(len(X)))
            X = pd.concat([ls,X],axis=1, ignore_index=True)
        df = pd.concat([X,y.rename("y")],axis=1, ignore_index=True)
        df_split = [df.iloc[i*batch_size:(i+1)*batch_size] for i in range(no_batch)] #divide entire dataframe into batches

        theta = np.ones(len(X.columns)).flatten() #initializing theta with zeros
        for i in range(n_iter):
            if lr_type=='inverse': # if type is inverse, lr becomes original lr / iteration number
                lr = learn_rate/(i+1)
            df2 = df_split[i%no_batch] #choose batch (X,y) for this iteration
            df2 = df2.reset_index(drop=True)
            X_iter = df2.iloc[:,0:len(df2.columns)-1]
            y_iter = df2.iloc[:,-1]
            self.X_new = X_iter
            self.y_new = y_iter
            xdash = X_iter.transpose()
            y_pred = np.dot(X_iter, theta) #calculating y_pred for entire matrix
            # loss = sigmoid(y_pred)-y_iter
            thet = np.array(theta) 
            gradient = grad(self.cost_func_L2)
            
            gradient_fin = gradient(thet)
            theta = theta - lr * gradient_fin/batch_size #updating theta            
        self.coef_= theta
        pass

    def fit_autograd(self, X, y, batch_size=1, n_iter=500, lr=0.001, lr_type='constant'):
        rows = len(X)
        no_batch = rows//batch_size # total number of batches

        learn_rate = lr
        if(self.fit_intercept): # add column of 1s in the beginning if fit intercept is true           
            ls = pd.Series(np.ones(len(X)))
            X = pd.concat([ls,X],axis=1, ignore_index=True)
        df = pd.concat([X,y.rename("y")],axis=1, ignore_index=True)
        df_split = [df.iloc[i*batch_size:(i+1)*batch_size] for i in range(no_batch)] #divide entire dataframe into batches

        theta = np.zeros(len(X.columns)).flatten() #initializing theta with zeros
        for i in range(n_iter):
            if lr_type=='inverse': # if type is inverse, lr becomes original lr / iteration number
                lr = learn_rate/(i+1)
            df2 = df_split[i%no_batch] #choose batch (X,y) for this iteration
            df2 = df2.reset_index(drop=True)
            X_iter = df2.iloc[:,0:len(df2.columns)-1]
            y_iter = df2.iloc[:,-1]
            self.X_new = X_iter
            self.y_new = y_iter
            xdash = X_iter.transpose()
            y_pred = np.dot(X_iter, theta) #calculating y_pred for entire matrix
            # loss = sigmoid(y_pred)-y_iter
            thet = np.array(theta) 
            gradient = grad(self.cost_func)
            gradient_fin = gradient(thet)
            gradient_fin =np.array([0 if math.isnan(i) else i for i in gradient_fin ])
            theta = theta - lr * gradient_fin/batch_size #updating theta            
        self.coef_= theta
        
        pass

    def fit_autograd_kclass(self, X, y, batch_size=1, n_iter=500, lr=0.001, lr_type='constant'):

        num_classes = len(np.unique(y))
        self.labels = np.array(list(set(y)))
        theta = np.zeros((len(X.columns),num_classes))
        rows = len(X)
        no_batch = rows//batch_size # total number of batches

        learn_rate = lr
        if(self.fit_intercept): # add column of 1s in the beginning if fit intercept is true           
            ls = pd.Series(np.ones(len(X)))
            X = pd.concat([ls,X],axis=1, ignore_index=True)
        df = pd.concat([X,y.rename("y")],axis=1, ignore_index=True)
        df_split = [df.iloc[i*batch_size:(i+1)*batch_size] for i in range(no_batch)] #divide entire dataframe into batches
        for i in range(n_iter):
            if lr_type=='inverse': # if type is inverse, lr becomes original lr / iteration number
                lr = learn_rate/(i+1)
            df2 = df_split[i%no_batch] #choose batch (X,y) for this iteration
            df2 = df2.reset_index(drop=True)
            X_iter = df2.iloc[:,0:len(df2.columns)-1]
            y_iter = df2.iloc[:,-1]
            self.X_new = X_iter
            self.y_new = y_iter
            gradient = grad(self.cross_entropy_multi)
            gradient_fin = gradient(theta)
            theta = theta - lr * gradient_fin/batch_size #updating theta            
        self.coef_= theta
        
        pass

    def fit_kclass(self, X, y, batch_size=1, n_iter=100, lr=0.01,lr_type='constant'):
        
        rows = len(X) 
        no_batch = rows//batch_size # total number of batches
        learn_rate = lr 
        if(self.fit_intercept): # add column of 1s in the beginning if fit intercept is true
            ls = pd.Series(np.ones(len(X))) 
            X = pd.concat([ls,X],axis=1, ignore_index=True)
        df = pd.concat([X,y.rename("y")],axis=1, ignore_index=True)
        df_split = [df.iloc[i*batch_size:(i+1)*batch_size] for i in range(no_batch)] #divide entire dataframe into batches
        X_iter = df.iloc[:,0:len(df.columns)-1] 
        y_iter = df.iloc[:,-1]
        num_classes = len(np.unique(y_iter))
        self.labels = np.array(list(set(y_iter)))
        self.theta = np.zeros((len(X_iter.columns),num_classes)) #initializing theta with random values
        for iteration in trange(n_iter):
            if lr_type=='inverse': # if type is inverse, lr becomes original lr / iteration number
                lr = learn_rate/(i+1)
            theta_copy = self.theta.copy() #defined so that all values of theta are updated simultaneously 
            df2 = df_split[iteration%no_batch] #choose batch (X,y) for this iteration
            df2 = df2.reset_index(drop=True)    
            X_iter = df2.iloc[:,0:len(df2.columns)-1] 
            y_iter = df2.iloc[:,-1]
            for cl in range(num_classes):
                cost=0  
                self.theta_copy = theta_copy
                identity = (y_iter == self.labels[cl]).astype(float)
                sm = self.softmax(X_iter,cl,self.theta)
                cost -= X_iter.T.dot(identity-sm)
                self.theta[:,cl] = self.theta[:,cl]-lr*cost/batch_size # updating theta according to prev theta values  
                  
        self.coef_= self.theta     
        pass

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        if(self.fit_intercept): # add column of 1s in the beginning if fit intercept is true  
            ls = pd.Series(np.ones(len(X)))
            X = pd.concat([ls,X],axis=1)
        y_hat_1 = pd.Series(np.dot(X,self.coef_).flatten()) # y = X*theta
        y_hat = pd.Series([1 if i>0  else 0 for i in y_hat_1])
        return y_hat
    
    def predict_kclass(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        if(self.fit_intercept): # add column of 1s in the beginning if fit intercept is true  
            ls = pd.Series(np.ones(len(X)))
            X = pd.concat([ls,X],axis=1, ignore_index=True)
        y_hat = np.zeros(len(X))
        num_classes=len(self.labels)
        P = np.zeros((len(X),num_classes))
        for i in range(self.coef_.shape[1]):
            P[:,i]= self.softmax(X, i,self.coef_)
    
        y_hat=self.labels[np.argmax(P,axis=1)]
        y_hat=pd.Series(y_hat)
        return y_hat

    def plot_surface(self, X, y,i,j):

        # Reference: https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/
        fig = plt.figure()
        c,x1,x2 = self.coef_[0],self.coef_[i],self.coef_[j]

        m = -x1/x2
        c /= -x2
        xmin, xmax, ymin, ymax = -5, 5, -1, 1
        Xs = np.array([xmin, xmax])
        ys = m*Xs + c
        plt.plot(Xs, ys, 'k', lw=1, ls='--')
        plt.fill_between(Xs, ys, ymin, color='tab:orange', alpha=0.2)
        plt.fill_between(Xs, ys, ymax, color='tab:blue', alpha=0.2)
        plt.scatter(X[y==0][0],X[y==0][1],s=8,alpha=0.5,cmap='Paired',label = "0")
        plt.scatter(X[y==1][0],X[y==1][1],s=8,alpha=0.5,cmap='Paired',label = "1")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.ylabel(r'$x_2$')
        plt.xlabel(r'$x_1$')
        plt.show()
        fig.savefig("./plots/Decision.png")

    def ret_theta(self):
        return self.coef_  