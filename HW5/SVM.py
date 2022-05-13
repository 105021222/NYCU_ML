from libsvm.svmutil import *
import numpy as np

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

X_train=load_x('X_train.csv')
Y_train=load_y('Y_train.csv')
X_test=load_x('X_test.csv')
Y_test=load_y('Y_test.csv')

mode={'linear kernel':'-t 0','polynomial kernel':'-t 1','RBF kernel':'-t 2'}
for m,parameter in mode.items():
    model=svm_train(Y_train,X_train,'-q '+parameter)
    p_label,p_acc,p_val=svm_predict(Y_test,X_test,model,'-q')
    print(m,'accuracy:{:.2f} %'.format(p_acc[0]))
    


