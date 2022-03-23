import numpy as np
import matplotlib.pyplot as plt
from lse import LSE
from newtonmethod import newton_method

def plot_curve(ax,coefficient):
    x_range=np.linspace(-6,6,num=100)
    fx_range=[]
    for i in range(len(x_range)):
        fx_value=0
        for j in range(n):
            fx_value+=coefficient[n-1-j,0]*x_range[i]**j
        fx_range.append(fx_value)
    ax.plot(x_range,fx_range)
    print('Fitting Line:',end='')
    for i in range(n-1):
        print(coefficient[i,0],'X^',n-1-i,end='')
        if(coefficient[i+1,0]>=0):
            print('+',end='')
        else:
            print('-')
    print(coefficient[-1,0])
    

x=[]
y=[]
fp=open('testfile.txt','r')
line=fp.readline()
while line:
    a,b=line.split(',')
    x.append(float(a))
    y.append(float(b))
    line=fp.readline()
fp.close()


while True:
    fig,(ax1,ax2)=plt.subplots(2,1)
    ax1.scatter(x,y,s=10,c='r')
    ax2.scatter(x,y,s=10,c='r')   

    x=np.asarray(x,dtype='float64')
    b=np.asarray(y,dtype='float64')
    n=int(input('Input polynomial bases n:'))
    Lambda=int(input('Input lambda:'))
    A=np.empty((len(x),n))
    for i in range(len(x)):
        for j in range(n):
            A[i,j]=x[i]**(n-1-j)
    b=np.resize(b,(len(b),1))

    coefficient,error=LSE(A,Lambda,b)
    print('\nLSE:')
    plot_curve(ax1,coefficient)
    print('Total error:',error,'\n')

    coefficient,error=newton_method(A,b)
    print('Newton\'s method:')
    plot_curve(ax2,coefficient)
    print('Total error:',error)
    plt.show()
    print('\n-----------')
    



    
