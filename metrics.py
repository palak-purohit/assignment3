import numpy as np
import pandas as pd
def accuracy(y_hat, y):

    assert(y_hat.size == y.size)
    # TODO: Write here
    y = y.reset_index(drop = True)
    y_hat = y_hat.reset_index(drop = True)
    # ctr = 0
    # for i in range(len(y)):
    #     if(y_hat[i]==y[i]):
    #         ctr+=1

    # print(ctr)
    # print("HELLOO")
    return np.mean(y == y_hat)

def precision(y_hat, y, cls):
    ctr=0.0
    y = y.reset_index(drop = True)
    y_hat = y_hat.reset_index(drop = True)
    df = pd.concat([y_hat, y], axis=1, ignore_index=True)
    counter = y_hat.value_counts().to_dict() 

    if cls not in y_hat.unique():
        return 0
    for i in range(len(y_hat)):
        if y_hat[i]==y[i]==cls:
            ctr+=1
    return ctr/counter[cls]

def recall(y_hat, y, cls):

    y = y.reset_index(drop = True)
    y_hat = y_hat.reset_index(drop = True)
    df = pd.concat([y_hat, y], axis=1, ignore_index=True)
    ctr=0.0
    counter = y.value_counts().to_dict()

    for i in range(len(y_hat)):
        if y_hat[i]==y[i]==cls:
            ctr+=1
    return ctr/counter[cls]
   
def rmse(y_hat, y):

    y = y.reset_index(drop = True)
    y_hat = y_hat.reset_index(drop = True)
    return (np.square(np.subtract(y_hat,y)).mean())**0.5

def mae(y_hat, y):
    y = y.reset_index(drop = True)
    y_hat = y_hat.reset_index(drop = True)
    return np.absolute(np.subtract(y_hat,y)).mean()
