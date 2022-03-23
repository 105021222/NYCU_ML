import numpy as np
from util import print_numbers

def test_discrete(images,labels,test_images,test_labels):
    image_number,row,column=images.shape
    bins=np.zeros((32,10,row,column),dtype=np.float32)
    for k in range(image_number):
        lable=labels[k]
        for i in range(row):
            for j in range(column):
                tag=images[k,i,j]//8
                bins[tag,lable,i,j]+=1
    for v in range(10):
        for i in range(row):
            for j in range(column):
                count=0
                for k in range(32):
                    count+=bins[k,v,i,j]
                bins[:,v,i,j]=(bins[:,v,i,j]+1)/(count+32)

    print_numbers(bins,16)

    error=0
    test_number,row,column=test_images.shape
    for k in range(test_number):
        posterior=np.zeros(10,dtype=np.float32)
        for i in range(28):
            for j in range(28):
                tag=test_images[k,i,j]//8
                for v in range(10):
                    posterior[v]+=np.log(bins[tag,v,i,j])
        posterior/=np.sum(posterior)
        print('Posterior(in log scale):')
        for v in range(10):
            print('{}:{}'.format(v,posterior[v]))
        print('Prediction:',posterior.argmin(),'Ans:',test_labels[k],'\n')
        if test_labels[k]!=posterior.argmin():
            error+=1
    print('Error rate:',error/test_number)

