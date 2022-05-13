import numpy as np
import matplotlib.pyplot as plt

def point_generate(m,v):
    x=np.random.uniform(0,1,12)
    x=np.sum(x)-6
    x=x*np.sqrt(v)+m
    return x

def data_generate(n,mx,vx,my,vy):
    D=np.zeros((n,2))
    for i in range(n):
        D[i,0]=point_generate(mx,vx)
        D[i,1]=point_generate(my,vy)
    return D

def get_matrix(n,D1,D2):
    A=np.ones((n*2,3))
    A[:n,1:]=D1
    A[n:,1:]=D2
    return A

def get_y(n):
    y=np.zeros((n*2,1))
    y[n:]=np.ones((n,1))
    return y

def visualize(ax,title,A,y,w):
    class0=[]
    class1=[]
    tp=fp=fn=tn=0
    for i in range(2*n):
        if A[i]@w>=0:
            class1.append(A[i,1:])
            if y[i,0]==1:
                tp+=1
            else:
                fp+=1
        else:
            class0.append(A[i,1:])
            if y[i,0]==0:
                tn+=1
            else:
                fn+=1
    class1 = np.array(class1)
    class0 = np.array(class0)
    ax.plot(class0[:,0],class0[:,1],'ro')
    ax.plot(class1[:,0],class1[:,1],'bo')
    ax.set_title(title)
    print(title,'\n')
    print('w:\n',w[0,0],'\n',w[1,0],'\n', w[2,0],'\n')
    print('Confusion Matrix:')
    print('\t\t\t Is cluster 1\t Is cluster 2')
    print('Predict cluster 1\t   ', tp, '\t\t   ', fn)
    print('Predict cluster 2\t   ', fp, '\t\t   ', tn)
    print('\nSensitivity (Successfully predict cluster 1): ', tp/(tp+fn))
    print('Specificity (Successfully predict cluster 2): ', tn/(tp+fn))

def gradient_descent(ax,A,y):
    w=np.random.rand(3,1)
    g_norm=1
    count=1
    while(g_norm>0.01 and count<100000):
        g=(A.T)@(y-(1/(1+np.exp(-A@w))))
        g_norm=np.sqrt(np.sum(g**2))
        w=w+0.1*g
        count+=1
    print(count)
    visualize(ax,'Gradient descent',A,y,w)

def newton(ax,A,y):
    w=np.random.rand(3,1)
    H=np.zeros((3,3))
    D=np.zeros((n*2,n*2))
    for i in range(n*2):
        D[i,i]=np.exp(-A[i]@w)/(1+np.exp(-A[i]@w))**2
    H=A.T@D@A
    g_norm=1
    count=0
    while(g_norm>0.01 and count<100000):
        if np.linalg.det(H)==0:
            g=(A.T)@(y-(1/(1+np.exp(-A@w))))
        else:
            H_inv=np.linalg.inv(H)
            g=H_inv@A.T@(y-1/(1+np.exp(-A@w)))
        g_norm=np.sqrt(np.sum(g**2))
        w=w+0.1*g
        count+=1
    print(count)
    visualize(ax,'Newton\'s method',A,y,w)



if __name__=='__main__':
    n=50
    mx1=1
    vx1=2
    my1=1
    vy1=2
    mx2=3
    vx2=4
    my2=3
    vy2=4

    D1=data_generate(n,mx1,vx1,my1,vy1)
    D2=data_generate(n,mx2,vx2,my2,vy2)
    A=get_matrix(n,D1,D2)
    y=get_y(n)
    
    fig,(ax0,ax1,ax2)=plt.subplots(1,3)
    ax0.plot(D1[:,0],D1[:,1],'ro')
    ax0.plot(D2[:,0],D2[:,1],'bo')
    ax0.set_title('Ground truth')
    gradient_descent(ax1,A,y)
    newton(ax2,A,y)
    plt.show()



    