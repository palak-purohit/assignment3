
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *
from tqdm import trange
np.random.seed(42)

from sklearn.datasets import load_breast_cancer
data,target = load_breast_cancer(return_X_y=True,as_frame=True)
data = (data - data.min( )) / (data.max( )-data.min( ))

df = pd.concat([data, target.rename("target")],axis=1, ignore_index=True)
# print(data)
rng = np.arange(len(df)) #shuffling the dataset
np.random.shuffle(rng)
df = df.iloc[rng].reset_index(drop=True)


n_folds = 3
net_accuracy = 0.0
accuracies=[]
size = len(data)//n_folds
folds_X = []
folds_y = []
for i in range(n_folds): #store data divided into n folds in folds_X and folds_Y
    folds_X.append(df.iloc[i*size:(i+1)*size].iloc[:,:-1])
    folds_y.append(df.iloc[i*size:(i+1)*size].iloc[:,-1])

for i in trange(n_folds):
    copy_X = folds_X.copy() 
    copy_y = folds_y.copy()
    X_test,y_test = copy_X[i],copy_y[i] #1 fold (ith one) used for testing
    copy_X.pop(i) # other folds exculding test one used for training
    copy_y.pop(i)
    X_train,y_train = pd.concat(copy_X), pd.concat(copy_y)
    
    LR = LogisticRegression(fit_intercept=True)
    LR.fit_vectorised(X_train.reset_index(drop=True), y_train.reset_index(drop=True),len(X_train.reset_index(drop=True)),1000,3)
    y_hat = LR.predict(X_test.reset_index(drop=True))
    net_accuracy+= accuracy(y_hat, y_test.reset_index(drop=True)) #net accuracy is average of all acccuracies
    accuracies.append(accuracy(y_hat, y_test.reset_index(drop=True)))

print("----WITHOUT AUTOGRAD----")
print("Total accuracy of the k-fold model is",net_accuracy/n_folds)
print("Accuracies of 3 folds",accuracies)

net_accuracy = 0.0
accuracies=[]

for i in trange(n_folds):
    copy_X = folds_X.copy() 
    copy_y = folds_y.copy()
    X_test,y_test = copy_X[i],copy_y[i] #1 fold (ith one) used for testing
    copy_X.pop(i) # other folds exculding test one used for training
    copy_y.pop(i)
    X_train,y_train = pd.concat(copy_X), pd.concat(copy_y)
    
    LR = LogisticRegression(fit_intercept=True)
    LR.fit_autograd(X_train.reset_index(drop=True), y_train.reset_index(drop=True),len(X_train.reset_index(drop=True)),1000,3)
    y_hat = LR.predict(X_test.reset_index(drop=True))
    net_accuracy+= accuracy(y_hat, y_test.reset_index(drop=True)) #net accuracy is average of all acccuracies
    accuracies.append(accuracy(y_hat, y_test.reset_index(drop=True)))

print("----WITH AUTOGRAD----")
print("Total accuracy of the k-fold model is",net_accuracy/n_folds)
print("Accuracies of 3 folds",accuracies)

N = 20
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series([1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1])


# for fit_intercept in [True, False]:
LR = LogisticRegression(fit_intercept=True)
LR.fit_vectorised(X, y,1,500,0.3) # here you can use fit_non_vectorised / fit_autograd methods
# y_hat = LR.predict(X)
LR.plot_surface(X, y,1,2)
