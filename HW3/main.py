import numpy as np
import matplotlib.pyplot as plt

def data_generate(m,s):
    x=np.random.uniform(0,1,12)
    x=np.sum(x)-6
    x=x*np.sqrt(s)+m
    return x

def seq_estimator(m_0,s_0):
    print("Data point source function: N({},{})".format(m_0,s_0))
    m=0
    s=0
    i=0
    while(abs(m-m_0)>1e-2 or abs(s-s_0)>1e-1):
        i+=1
        x=data_generate(m_0,s_0)
        s=(s*(i-1)+x**2+m**2*(i-1)-(m*(i-1)+x)**2/i)/i
        m=(m*(i-1)+x)/i
        print("Add data point:",x)
        print("Mean = {} Var = {}".format(m,s))
    print(i)

def poly_basis_generate(n,a,w):
    y=0
    x=np.random.uniform(-1,1)
    e=data_generate(0,a)
    for i in range(n):
        y+=w[i]*x**i
    y+=e
    return x,y

def plot(num_points,mean,var,title):
    t=np.linspace(-2,2,500)
    mean_predict=np.zeros(500)
    var_predict=np.zeros(500)
    for i in range(500):
        X=np.asarray([t[i]**k for k in range(n)])
        mean_predict[i]=(X@mean).item(0)
        var_predict[i]=(a+X@var@X.T).item(0)
        
    plt.plot(x_value[0:num_points],y_value[0:num_points],'bo')
    plt.plot(t,mean_predict,'k-')
    plt.plot(t,mean_predict+var_predict,'r-')
    plt.plot(t,mean_predict-var_predict,'r-')
    plt.xlim(-2,2)
    plt.ylim(-20,20)
    plt.title(title)
    plt.show()

if __name__=='__main__':
    b=1
    n=3
    a=3
    w=np.asarray([1,2,3])

    Num=1000
    x_value=[]
    y_value=[]
    mean_value=[]
    var_value=[]

    mean=np.zeros((n,1))
    var=(1/b)*np.identity(n)
    i=1
    while(i<=Num):
        x,y=poly_basis_generate(n,a,w)
        print("Add data point:({},{})\n".format(x,y))

        X=np.asarray([x**i for i in range(n)]).reshape((1,-1))
        S=np.linalg.inv(var)
        post_var=np.linalg.inv((1/a)*X.T@X+S)
        post_mean=post_var@((1/a)*X.T*y+S@mean)
        print("posterior mean:")
        print(post_mean,"\n")
        print("posterior var:")
        print(post_var,"\n")

        predict_mean=(X@post_mean).item(0)
        predict_var=(a+X@post_var@X.T).item(0)
        print("Predictive distribution ~ N({:.5f},{:.5f})".format(predict_mean,predict_var))
        print("---")

        x_value.append(x)
        y_value.append(y)
        if i==10 or i==50 or i==Num:
            mean_value.append(post_mean)
            var_value.append(post_var)
        mean=post_mean
        var=post_var
        i+=1

    plot(10,mean_value[0],var_value[0],"After 10 incomes")
    plot(50,mean_value[1],var_value[1],"After 50 incomes")
    plot(Num,mean_value[2],var_value[2],"Predict result")
    plot(0,w,np.zeros((n,n)),"Ground truth")