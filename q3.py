import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *

import warnings

warnings.filterwarnings("ignore",category =RuntimeWarning)

np.random.seed(42)

from sklearn.datasets import load_digits
data,target = load_digits(return_X_y=True,as_frame=True)
df = pd.concat([data, target.rename("target")],axis=1, ignore_index=True)

n_folds = 4
net_accuracy = 0.0
accuracies=[]
size = len(data)//n_folds
folds_X = []
folds_y = []
for i in range(n_folds): #store data divided into n folds in folds_X and folds_Y
    folds_X.append(df.iloc[i*size:(i+1)*size].iloc[:,:-1])
    folds_y.append(df.iloc[i*size:(i+1)*size].iloc[:,-1])

for i in range(n_folds):
    copy_X = folds_X.copy() 
    copy_y = folds_y.copy()
    X_test,y_test = copy_X[i],copy_y[i] #1 fold (ith one) used for testing
    copy_X.pop(i) # other folds exculding test one used for training
    copy_y.pop(i)
    X_train,y_train = pd.concat(copy_X).reset_index(drop=True), pd.concat(copy_y).reset_index(drop=True)
    # print(X_train)
    LR = LogisticRegression(fit_intercept=True)
    LR.fit_kclass(X_train.reset_index(drop=True), y_train.reset_index(drop=True),len(X_train.reset_index(drop=True)),100,0.1)
    y_hat = LR.predict_kclass(X_test.reset_index(drop=True))
    # print(y_hat)
    net_accuracy+= accuracy(y_hat, y_test.reset_index(drop=True)) #net accuracy is average of all acccuracies
    accuracies.append(accuracy(y_hat, y_test.reset_index(drop=True)))

print("Total accuracy of the k-fold model is",net_accuracy/n_folds)
print("Accuracies of 3 folds",accuracies)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_hat))

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(data)
fig = plt.figure()
plt.scatter(proj[:, 0], proj[:, 1], c=target, cmap="Paired")
plt.colorbar()
# plt.show()

fig.savefig("./plots/PCA.png")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import plot_confusion_matrix
# cm = confusion_matrix(test_generator.classes, y_pred)
# # or
# #cm = np.array([[1401,    0],[1112, 0]])

# plt.imshow(cm, cmap=plt.cm.Blues)
# plt.xlabel("Predicted labels")
# plt.ylabel("True labels")
# plt.xticks([], [])
# plt.yticks([], [])
# plt.title('Confusion matrix ')
# plt.colorbar()
# plt.show()

# import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    # plt.show()
    
    fig.savefig("./plots/Confusion.png")


plot_confusion_matrix ( confusion_matrix(y_test, y_hat), 
                      normalize    = False,
                      target_names = ['0', '1', '2','3','4','5','6','7','8','9'],
                      title        = "Confusion Matrix")