from re import L
import numpy as np
import matplotlib.pyplot as plt
import gzip
import math

def read_images(filename):
    fp=gzip.open(filename,'r')
    magic_number=int.from_bytes(fp.read(4),byteorder='big')
    numbers=int.from_bytes(fp.read(4),byteorder='big')
    rows=int.from_bytes(fp.read(4),byteorder='big')
    columns=int.from_bytes(fp.read(4),byteorder='big')
    
    buf=fp.read(rows*columns*numbers)
    images=np.frombuffer(buf,dtype=np.uint8)
    images=images.reshape(numbers,rows*columns)

    return images


def read_labels(filename):
    fp=gzip.open(filename,'r')
    magic_number=int.from_bytes(fp.read(4),byteorder='big')
    numbers=int.from_bytes(fp.read(4),byteorder='big')
    
    buf=fp.read(numbers)
    labels=np.frombuffer(buf,dtype=np.uint8)
    
    return labels


def predict(image,bins):
    postirior=np.zeros(10,dtype=np.float32)
    for i in range(28):
        for j in range(28):
            tag=image[i,j]//8
            for k in range(10):
                postirior[k]+=np.log(bins[tag,k,i,j])

    n=9
    for i in range(9):
        if postirior[i]>postirior[n]:
            n=i

    return postirior,n



def test_discrete(images,labels,test_images,test_labels):
    image_number,row,column=images.shape
    bins=np.zeros((32,10,row,column),dtype=np.float32)
    for k in range(image_number):
        value=labels[k]
        for i in range(row):
            for j in range(column):
                tag=images[k,i,j]//8
                bins[tag,value,i,j]+=1
    for value in range(10):
        for i in range(row):
            for j in range(column):
                count=0
                for k in range(32):
                    count+=bins[k,value,i,j]
                bins[:,value,i,j]=(bins[:,value,i,j]+1)/(count+32)

    print_numbers(bins,16)

    error=0
    test_number,row,column=test_images.shape
    for i in range(100):
        postirior=np.zeros(10,dtype=np.float32)
        for i in range(28):
            for j in range(28):
                tag=image[i,j]//8
                for k in range(10):
                    postirior[k]+=np.log(bins[tag,k,i,j])
        postirior/=np.sum(postirior)
        print('Postirior(in log scale):')
        for v in range(10):
            print('{}:{}'.format(v,postirior[v]))
        print('Prediction:',postirior.argmin(),'Ans:',test_labels[i],'\n')
        if test_labels[i]!=postirior.argmin():
            error+=1
    print('Error rate:',error/100)

def get_mean_var_vprob(images,labels):
    image_number,row,column=images.shape
    mean=np.zeros((10,row,column),dtype=np.float32)
    var=np.zeros((10,row,column),dtype=np.float32)
    total=np.zeros(10,dtype=np.float32)
    for k in range(image_number):
        mean[labels[k],:,:]+=images[k,:,:]
        total[labels[k]]+=1
    for k in range(10):
        mean[k,:,:]/=total[k]
    for k in range(image_number):
        var[labels[k],:,:]+=(images[k,:,:]-mean[labels[k],:,:])**2
    for k in range(10):
        var[k,:,:]/=total[k]
    total/=np.sum(total)
    
    return mean,var,total


def test_continue(images,labels,test_images,test_labels):
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
    for k in range(10000):
        postirior=np.zeros(10,dtype=np.float32)
        for v in range(10):
            for i in range(row):
                for j in range(column):
                    postirior[v]+=prob[test_images[k,i,j],v,i,j]
        postirior[v]+=np.log(v_prob[v])
        postirior/=np.sum(postirior)
        '''print(postirior)
        print(postirior.argmin(),test_labels[k])'''
        print('Postirior(in log scale):')
        for v in range(10):
            print('{}:{}'.format(v,postirior[v]))
        print('Prediction:',postirior.argmin(),'Ans:',test_labels[k],'\n')
        if postirior.argmin()!=test_labels[k]:
            error+=1
    print('Error rate:',error/10000)

def print_numbers(prob,x):
    print('Imagination of numbers in Bayesian classifier:')
    for v in range(10):
        print(v,':')
        for i in range(28):
            for j in range(28):
                print('1' if np.argmax(prob[:,v,i,j])>=x else '0',end=' ')
            print()
        print()
    print()


def E_Step(images,v_prob,p,Z):
	for k in range(60000):
		temp = np.zeros(10)
		for v in range(10):
			x=1
			for i in range(28*28):
				if images[k,i]:
					x*=p[v,i]
				else:
					x*=(1-p[v,i])
			temp[v] = v_prob[v]*x
		if np.sum(temp)==0:
			Z[k]=temp
		else:
			Z[k]=temp/np.sum(temp)
	return Z

def M_step(images,v_prob,p,Z):
    N=np.sum(Z,axis=0)
    for i in range(28*28):
        for v in range(10):
            if N[v]==0:
                N[v]=1
            p[v,i]=np.dot(images[:,i],Z[:,v])/N[v]
    for v in range(10):
        v_prob[v]=N[i]/60000
    return v_prob,p
images=read_images('train-images-idx3-ubyte.gz')
labels=read_labels('train-labels-idx1-ubyte.gz')
test_images=read_images('t10k-images-idx3-ubyte.gz')
test_labels=read_labels('t10k-labels-idx1-ubyte.gz')


'''test_continue(images,labels,test_images,test_labels)'''
'''test_discrete(images,labels,test_images,test_labels)'''


image_number=60000
v_prob=np.full(10,0.1)
p=np.random.rand(10,28*28)
p_old=np.zeros((10,28*28))
Z = np.full((image_number,10), 0.1)
E_Step(images,v_prob,p,Z)
print(Z)




            



'''for i in range(1):
    tmp=images[i,:,:]
    plt.imshow(tmp)
    plt.show()
'''