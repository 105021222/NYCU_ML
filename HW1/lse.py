import numpy as np
from matrix import inverse


def LSE(A,Lambda,b):
    m,n=A.shape
    x=inverse(A.T@A+Lambda*np.identity(n))@A.T@b
    p,q=b.shape
    error=0
    for i in range(p):
        error+=(A@x-b)[i,0]**2
    return x,error
