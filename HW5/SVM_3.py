from libsvm.svmutil import *
import numpy as np
from scipy.spatial.distance import cdist

def load_x(filename):
    x=[]
    f = open(filename, 'r')
    for line in f.readlines():
        tmp=[]
        for i in line.split(','):
            tmp.append(float(i))
        x.append(tmp)
    f.close()
    x=np.asarray(x)
    return x

def load_y(filename):
    y=[]
    f = open(filename, 'r')
    for line in f.readlines():
        y.append(float(line))
    f.close()
    y=np.asarray(y)
    return y

def linearKernel(X1, X2):
    kernel = X1 @ X2.T
    return kernel
    
def RBFKernel(X1, X2, gamma):
    dist=cdist(X1, X2, metric='sqeuclidean')
    kernel = np.exp((-1 * gamma * dist))
    return kernel

def user_kernel(X1,X2,gamma):
    k=np.hstack((np.arange(1,len(X1)+1).reshape(-1,1),linearKernel(X1, X2)+RBFKernel(X1, X2, gamma)))
    return k

X_train=load_x('X_train.csv')
Y_train=load_y('Y_train.csv')
X_test=load_x('X_test.csv')
Y_test=load_y('Y_test.csv')

k=user_kernel(X_train,X_train,0.125)
k_star=user_kernel(X_test,X_train,0.125)

#-t 4:precomputed kernel 
model=svm_train(Y_train,k,'-t 4 -q')
svm_predict(Y_test,k_star,model,)