import numpy as np

def LU_decomposition(A):
    m,n=A.shape
    U=A.copy()
    P=np.identity(n)
    L=np.identity(n)
    for i in range(n):
        pivot=U[i,i]
        pivot_Row=i
        
        for k in range(i+1,n):
            if pivot==0:
                pivot=U[k,i]
                pivot_Row=k
        #swap pivot_Row and i-th Row
        if pivot!=0:
            U[[i,pivot_Row],:]=U[[pivot_Row,i],:]
            P[[i,pivot_Row], :] = P[[pivot_Row, i], :]
            L[[i,pivot_Row],:]=L[[pivot_Row,i],:]
            L[:,[i,pivot_Row]]=L[:,[pivot_Row,i]]
            
        #eliminate
        for k in range(i+1,n):
            q=U[k,i]/U[i,i]
            U[k,i:]=U[k,i:]-q*U[i,i:]
            L[k:,i]=L[k:,i]+q*L[k:,k]
    
    return P,L,U
    

def L_inverse(L):
    m,n=L.shape
    L_inv=np.identity(n)
    for i in range(n-1):
        for k in range(i+1,n):
            L_inv[k,:]-=L_inv[i,:]*L[k,i]
    return L_inv

def U_inverse(U):
    m,n=U.shape
    D=np.identity(n)
    U_inv=np.identity(n)
    D_inv=np.identity(n)
    for i in range(n):
        D[i,i]=U[i,i]
        U[i,i:]=U[i,i:]/D[i,i]
    D_inv=D.copy()
    for i in range(n):
        D_inv[i,i]=1/D[i,i]
    for i in range(n-1,0,-1):
        for k in range(i-1,-1,-1):
            U_inv[k,:]-=U_inv[i,:]*U[k,i]
    return U_inv@D_inv

def inverse(A):
    m,n=A.shape
    P,L,U=LU_decomposition(A)
    L_inv=L_inverse(L)
    U_inv=U_inverse(U)
    return U_inv@L_inv@P.T


