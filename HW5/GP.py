import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def read_data(fliename):
    x=[]
    y=[]
    f = open(fliename, 'r')
    for line in f.readlines():
        point = line.split(' ')
        x.append(float(point[0]))
        y.append(float(point[1]))
    f.close()
    x=np.asarray(x)
    y=np.asarray(y)
    return x,y

def kernel(X1,X2,alpha,length_scale,sigma):
    #X1: (n) ndarray
    #X2: (m) ndarray
    #return (n,m)  ndarray
    kernel=(sigma**2)*(1+(X1.reshape(-1,1)-X2.reshape(1,-1))**2/(2*alpha*length_scale**2))**(-alpha)
    return kernel

def GP(X,y,beta,alpha,length_scale,sigma):
    C=kernel(X,X,alpha,length_scale,sigma)+(1/beta)*np.identity(len(X)) 
    X_test=np.linspace(-60,60,num=1000)
    mean_of_x_test=kernel(X,X_test,alpha,length_scale,sigma).T@np.linalg.inv(C)@y
    k_star=kernel(X_test,X_test,alpha,length_scale,sigma)+(1/beta)*np.identity(len(X_test))
    var_of_x_test=k_star-kernel(X,X_test,alpha,length_scale,sigma).T@np.linalg.inv(C)@kernel(X,X_test,alpha,length_scale,sigma)

    mean_of_x_test=mean_of_x_test.reshape(-1)
    sd_of_x_test=np.sqrt(np.diag(var_of_x_test))
    interval=1.96*sd_of_x_test

    plt.plot(X_test,mean_of_x_test,'k-')
    plt.fill_between(X_test,mean_of_x_test+interval,mean_of_x_test-interval,alpha=0.3,color='red')
    plt.scatter(X,y)
    plt.title('alpha: {:.5f}, length_scale: {:.5f}, sigma: {:.5f}'.format(alpha,length_scale,sigma))
    plt.xlim(-60, 60)
    plt.show()

def negative_log_likelihood(theta):
    #theta[0]: alpha
    #theta[1]: length_scale
    #theta[2]: sigma
    k=kernel(X,X,theta[0],theta[1],theta[2])
    C=k+(1/beta)*np.identity(len(X))
    negative_log_likelihood=(1/2)*(np.log(np.linalg.det(C))+y.T@np.linalg.inv(C)@y+len(C)*np.log(2*np.pi))
    return negative_log_likelihood


X,y=read_data("input.data")
beta=5
alpha=1
length_scale=1
sigma=1
GP(X,y,beta,alpha,length_scale,sigma)

#optimize
theta=[1,1,1] #initial parameter
res=minimize(negative_log_likelihood,theta)
alpha=res.x[0]
length_scale=res.x[1]
sigma=res.x[2]
GP(X,y,beta,alpha,length_scale,sigma)





