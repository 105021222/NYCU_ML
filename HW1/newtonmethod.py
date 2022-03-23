import numpy as np
from matrix import inverse

def newton_method(A,b):
    m,n=A.shape
    x0=np.random.randint(2,size=(n,1))
    delta=1
    error=0
    while delta>1e-6:
        delta=0
        x1=x0-inverse(2*A.T@A)@(2*A.T@A@x0-2*A.T@b)
        for i in range(n):
            delta+=(x1-x0)[i,0]**2
        x0=x1
    for i in range(m):
        error+=(A@x0-b)[i,0]**2
    return x0,error
    
        
    
