import numpy as np
import math
from util import print_numbers

def test_continuous(images,labels,test_images,test_labels):
    image_number,row,column=images.shape
    test_number,row,column=test_images.shape

    mean,var,v_prob=get_mean_var_vprob(images,labels)
    prob=np.zeros((256,10,28,28),dtype=np.float32)
    for k in range(256):
        for v in range(10):
            for i in range(28):
                for j in range(28):
                    if var[v,i,j]==0:
                        var[v,i,j]=1e3
                    prob[k,v,i,j]=(-(k-mean[v,i,j])**2)/(2*var[v,i,j])-np.log(math.sqrt(var[v,i,j]))

    print_numbers(prob,128)

    error=0
    for k in range(test_number):
        posterior=np.zeros(10,dtype=np.float32)
        for v in range(10):
            for i in range(row):
                for j in range(column):
                    posterior[v]+=prob[test_images[k,i,j],v,i,j]
        posterior[v]+=np.log(v_prob[v])
        posterior/=np.sum(posterior)
        print('Posterior(in log scale):')
        for v in range(10):
            print('{}:{}'.format(v,posterior[v]))
        print('Prediction:',posterior.argmin(),'Ans:',test_labels[k],'\n')
        if posterior.argmin()!=test_labels[k]:
            error+=1
    print('Error rate:',error/test_number)


def get_mean_var_vprob(images,labels):
    image_number,row,column=images.shape
    mean=np.zeros((10,row,column),dtype=np.float32)
    var=np.zeros((10,row,column),dtype=np.float32)
    total=np.zeros(10,dtype=np.float32)
    for k in range(image_number):
        mean[labels[k],:,:]+=images[k,:,:]
        total[labels[k]]+=1
    for v in range(10):
        mean[v,:,:]/=total[v]
    for k in range(image_number):
        var[labels[k],:,:]+=(images[k,:,:]-mean[labels[k],:,:])**2
    for v in range(10):
        var[v,:,:]/=total[v]
    total/=np.sum(total)
    
    return mean,var,total
