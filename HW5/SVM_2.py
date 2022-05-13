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

def grid_search(X_train,Y_train):
    optimal_opt=''
    optimal_acc=0
    cost=[0.125,0.25,0.5,1,2,4,8]
    mode=['linear kernel','polynomial kernel','RBF kernel']
    
    for m in mode:
        for c in cost:
            if m=='linear kernel':
                opt=f'-t 0 -c {c} -v 3 -q'
                print(m,f': cost:{c}')
                cv_acc=svm_train(Y_train,X_train,opt)
                if cv_acc>optimal_acc:
                    optimal_acc=cv_acc
                    optimal_opt=opt
            elif m=='polynomial kernel':
                gamma=[0.125,0.25,0.5,1,2,4,8]
                degree=[2,3,4]
                for g in gamma:
                    for d in degree:
                        opt=f'-t 1 -c {c} -g {g} -d {d} -v 3 -q'
                        print(m,f': cost:{c}, gamma:{g}, degree{d}')
                        cv_acc=svm_train(Y_train,X_train,opt)
                        if cv_acc>optimal_acc:
                            optimal_acc=cv_acc
                            optimal_opt=opt
            else:
                gamma=[0.125,0.25,0.5,1,2,4,8]
                for g in gamma:
                    opt=f'-t 2 -c {c} -g {g} -v 3 -q'
                    print(m,f': cost:{c}, gamma:{g}')
                    cv_acc=svm_train(Y_train,X_train,opt)
                    if cv_acc>optimal_acc:
                        optimal_acc=cv_acc
                        optimal_opt=opt
    return optimal_opt,optimal_acc

X_train=load_x('X_train.csv')
Y_train=load_y('Y_train.csv')
X_test=load_x('X_test.csv')
Y_test=load_y('Y_test.csv')
print(grid_search(X_train,Y_train))

#model=svm_train(Y_train,X_train,'-t 1 -c 4 -g 4 -d 2 -q -v 3')
#p_label,p_acc,p_val=svm_predict(Y_test,X_test,model,'-q')
#print(p_acc)
