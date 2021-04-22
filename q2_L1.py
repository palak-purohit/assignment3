import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *
from tqdm import trange
# import warnings

# warnings.filterwarnings("ignore",category =RuntimeWarning)
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
max_acc = 0
accuracies=[]

size = len(data)//n_folds
folds_X = []
folds_y = []
for i in range(n_folds): #store data divided into n folds in folds_X and folds_Y
    folds_X.append(df.iloc[i*size:(i+1)*size].iloc[:,:-1])
    folds_y.append(df.iloc[i*size:(i+1)*size].iloc[:,-1])

valid_folds = 3 # 3 folds for valid-train 
ld = 0
best_ld = [] #storing the 5 best lds
for i in trange(n_folds):
    copy_X = folds_X.copy()
    copy_y = folds_y.copy()
    X_test,y_test = copy_X[i],copy_y[i] #1 fold (ith one) used for testing
    copy_X.pop(i) # other folds exculding test one used for training
    copy_y.pop(i)
    X_train,y_train = pd.concat(copy_X), pd.concat(copy_y) 
    xtrainfolds=[]
    ytrainfolds=[]
    
    for j in range(valid_folds):
        size = len(X_train)//valid_folds #making valid folds
        xtrainfolds.append(X_train.iloc[j*size:(j+1)*size]) 
        ytrainfolds.append(y_train.iloc[j*size:(j+1)*size])
    for m in range(1,5):
        #varying ld
        ld_accuracy=0.0
        for k in range(valid_folds):
            
            xtraincpy = xtrainfolds.copy() #1 fold (ith one) used for validating
            ytraincpy = ytrainfolds.copy() # other folds exculding valid one used for training
            xvalid,yvalid = xtrainfolds[k],ytrainfolds[k]
            xtraincpy.pop(k)
            ytraincpy.pop(k)
            xtrainmini,ytrainmini =  pd.concat(xtraincpy), pd.concat(ytraincpy)            
            LR = LogisticRegression(fit_intercept=False)
            LR.fit_L1(xtrainmini.reset_index(drop=True), ytrainmini.reset_index(drop=True),len(xtrainmini.reset_index(drop=True)),200,0.5,m)
            y_hat = LR.predict(xvalid.reset_index(drop=True))
            acc = accuracy(y_hat,yvalid.reset_index(drop=True))
            ld_accuracy+=acc
        avg_accuracy = ld_accuracy/valid_folds #average accuracy for this ld
        if(avg_accuracy>max_acc):
            max_acc=avg_accuracy #storing max accuracy 
            ld = m
    best_ld.append(ld)


print("Optimal ld for all test-train folds is",best_ld)
print(max_acc)

thetas=[]
m = 0.1
ld=[]
for i in trange(1,400):
    #varying ld
    
    LR = LogisticRegression(fit_intercept=True)
    LR.fit_L1(data.iloc[:,:9], target,len(xtrainmini.reset_index(drop=True)),200,0.5,m)
    thetas.append(LR.ret_theta())
    m+=0.1
    ld.append(m)

count = 0
for i in np.array(thetas).transpose():
    plt.plot(ld,i,label=str(data.columns[count])) # plot for each n with legend
    count+=1

# plt.plot(ld,thetas) 
# naming the x axis 
plt.xlabel('lambda_val') 
# naming the y axis 
plt.ylabel('theta') 
# giving a title to my graph 
plt.title('theta vs lambda')
plt.legend(loc='upper right') 
# function to show the plot 
plt.savefig("./plots/q2L1_plot.png")

plt.show()