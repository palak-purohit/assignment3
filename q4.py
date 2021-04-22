import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
import time
from tqdm import trange

np.random.seed(42)

N = 500
P = 300

n_varyLR = []
n_varyN = []
p_varyLR = []
p_varyN = []
i_varyLR = []
i_varyN = []

for n in trange(1,500): # varying number of examples
    X = pd.DataFrame(np.random.randn(n, P))
    y = pd.Series(np.random.randint(2, size=n))

    begin = end = 0
    for i in range(5):
        LR = LogisticRegression()
        begin += time.time() #start and end for Logistic regression fitting
        LR.fit_vectorised(X, y,len(y),20) 
        end += time.time()
    begin /= 5
    end /= 5

    begin2 = end2 = 0
    for i in range(5):
        begin2 += time.time() #start and end for normal method
        LR.predict(X) 
        end2 += time.time()

    begin2 /= 5
    end2 /= 5

    n_varyLR.append(end-begin)
    n_varyN.append(end2-begin2)


for p in trange(1,300): # varying number of features
    X = pd.DataFrame(np.random.randn(N, p))
    y = pd.Series(np.random.randint(2, size=N))

    begin = end = 0
    for i in range(5):
        LR = LogisticRegression()
        begin += time.time() #start and end for Logistic regression fitting
        LR.fit_vectorised(X, y,len(y),20) 
        end += time.time()
    begin /= 5
    end /= 5

    begin2 = end2 = 0
    for i in range(5):
        begin2 += time.time() #start and end for normal method
        LR.predict(X) 
        end2 += time.time()

    begin2 /= 5
    end2 /= 5

    p_varyLR.append(end-begin)
    p_varyN.append(end2-begin2)

for iter in trange(1,100): # varying maximum
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randint(2, size=N))

    
    LR = LogisticRegression()
    begin = time.time() #start and end for Logistic regression fitting
    LR.fit_vectorised(X, y,len(y),iter) 
    end = time.time()

    begin2 = end2 = 0
    for i in range(5):

        begin2 += time.time() #start and end for normal method
        LR.predict(X) 
        end2 += time.time()

    begin2 /= 5
    end2 /= 5

    i_varyLR.append(end-begin)
    i_varyN.append(end2-begin2)

N = range(1,500)
p = range(1,300)
i = range(1,100)

fig1= plt.figure(1)

plt.plot(N,n_varyLR)
plt.plot(N,n_varyN)
plt.legend(["LR time", "Predict time"])
plt.xlabel("N")

fig2= plt.figure(2)

plt.plot(p,p_varyLR)
plt.plot(p,p_varyN)
plt.legend(["LR time", "Predict time"])
plt.xlabel("P")

fig3= plt.figure(3)

plt.plot(i,i_varyLR)
plt.plot(i,i_varyN)
plt.legend(["LR time", "Predict time"])
plt.xlabel("i")

fig1.savefig("./plots/Nvary.png")
fig2.savefig("./plots/Pvary.png")
fig3.savefig("./plots/ivary.png")

plt.show()