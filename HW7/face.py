from PIL import Image
import numpy as np
import os, re, sys
from scipy.spatial.distance import cdist
from datetime import datetime
import matplotlib.pyplot as plt

SHAPE=(60,60)

def read_pgm(filename):
    image=Image.open(filename)
    image=image.resize(SHAPE,Image.ANTIALIAS)
    image=np.array(image)
    label=int(re.findall('\d\d', filename)[0])
    return image.ravel().astype(np.float64),label

def read_data(dir):            
    data=[]
    labels=[]
    for filename in os.listdir(dir):
        image,lable = read_pgm(f'{dir}/{filename}')
        data.append(image)
        labels.append(lable)
    return np.asarray(data), np.asarray(labels)

def PCA(X,k):
    mean=np.mean(X,axis=0)
    cov=(X-mean)@(X-mean).T
    eigen_val,eigen_vec = np.linalg.eigh(cov)
    eigen_vec=(X-mean).T@eigen_vec
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:,i]=eigen_vec[:,i]/np.linalg.norm(eigen_vec[:,i])
    idx=np.argsort(eigen_val)[::-1]
    W=eigen_vec[:,idx][:,:k].real
    return W,mean

def LDA(X,labels,k):
    d=X.shape[1]
    labels=np.asarray(labels)
    c=np.unique(labels)
    mean=np.mean(X, axis=0)
    S_w=np.zeros((d, d))
    S_b=np.zeros((d, d))
    for i in c:
        X_i=X[np.where(labels==i)[0],:]
        mean_i=np.mean(X_i, axis=0)
        S_w+=(X_i-mean_i).T@(X_i-mean_i)
        S_b+=X_i.shape[0]*((mean_i - mean).T @ (mean_i - mean))
    eigen_val,eigen_vec=np.linalg.eig(np.linalg.pinv(S_w)@S_b)
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i]/np.linalg.norm(eigen_vec[:, i])
    idx=np.argsort(eigen_val)[::-1]
    W=eigen_vec[:,idx][:,:k].real
    return W

def linear_Kernel(X):
    return X@X.T

def polynomial_Kernel(X,r,c,d):
    return np.power(r*(X@X.T)+c,d)

def RBF_Kernel(X,r):
    return np.exp(-r*cdist(X,X,'sqeuclidean'))

def kernel_PCA(kernel,k):
    n=kernel.shape[0]
    one = np.ones((n, n), dtype=np.float64)/n
    kernel=kernel-one@kernel-kernel@one+one@kernel@one
    eigen_val,eigen_vec=np.linalg.eigh(kernel)
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / np.linalg.norm(eigen_vec[:, i])
    idx=np.argsort(eigen_val)[::-1]
    W=eigen_vec[:, idx][:, :k].real
    return kernel@W

def kernel_LDA(kernel,labels,k):
    labels=np.asarray(labels)
    c=np.unique(labels)
    n=kernel.shape[0]
    mean=np.mean(kernel,axis=0)
    M=np.zeros((n,n))
    N=np.zeros((n,n))
    for i in c:
        K_i=kernel[np.where(labels==i)[0],:]
        l=K_i.shape[0]
        mean_i=np.mean(K_i, axis=0)
        N+=K_i.T@(np.eye(l)-(np.ones((l, l),dtype=np.float64)/l))@K_i
        M+=l*((mean_i - mean).T@(mean_i - mean))
    eigen_val,eigen_vec=np.linalg.eig(np.linalg.pinv(N)@M)
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:,i]=eigen_vec[:, i]/np.linalg.norm(eigen_vec[:,i])
    idx=np.argsort(eigen_val)[::-1]
    W=eigen_vec[:,idx][:, :k].real
    return kernel@W

def visualize(x,title,W,mean=None):
    if mean is None:
        mean=np.zeros(x.shape[1])
    z=(x-mean)@W
    new_x=z@W.T+mean
    if not os.path.isdir(f"./{title}"):
        os.mkdir(title)
    #eigenface & fisherface
    if W.shape[1]==25:
        plt.clf()
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                plt.subplot(5, 5, idx + 1)
                plt.imshow(W[:, idx].reshape(SHAPE[::-1]), cmap='gray')
                plt.axis('off')
        plt.savefig(f'./{title}/{title}.png')
    #reconstruction
    if x.shape[0]==10:
        plt.clf()
        for i in range(2):
            for j in range(10):
                if i==1:
                    plt.subplot(2,10,j+1)
                    plt.imshow(x[j].reshape(SHAPE[::-1]), cmap='gray')
                    plt.axis('off')
                else:
                    plt.subplot(2,10,j+11)
                    plt.imshow(new_x[j].reshape(SHAPE[::-1]), cmap='gray')
                    plt.axis('off')
        plt.savefig(f'./{title}/reconstruction.png')
    

def recognition(train,train_labels,test,test_labels):
    dist=[]
    for i in range(test.shape[0]):
        i_dist=[]
        for j in range(train.shape[0]):
            i_dist.append((np.sum((train[j]-test[i])**2),train_labels[j]))
        i_dist.sort(key=lambda x: x[0])
        dist.append(i_dist)
    for k in [3,4,5,6,7]:
        correct=0
        total=test.shape[0]
        for i in range(test.shape[0]):
            i_dist=dist[i]
            neighbor=np.asarray([x[1] for x in i_dist[:k]])
            neighbor,count=np.unique(neighbor,return_counts=True)
            predict=neighbor[np.argmax(count)]
            if predict==test_labels[i]:
                correct+=1
        print(f'K={k},accuracy:{correct / total:>.3f}')
    print()



train,train_labels=read_data('./Yale_Face_Database/Training')
test,test_labels=read_data('./Yale_Face_Database/Testing')
data=np.vstack((train,test))
labels=np.hstack((train_labels,test_labels))

# part1
print('part1:')
random_idx=np.random.choice(data.shape[0],10)
random_data=data[random_idx]
#PCA
W,mean=PCA(data,25)
visualize(random_data,'PCA_eigenface',W,mean)
#LDA
W=LDA(data,labels,25)
visualize(random_data,'LDA_fisherface',W)

#part2
print('part2:')
W,mean=PCA(data,25)
train_proj=(train-mean)@W
test_proj=(test-mean)@W
recognition(train_proj,train_labels,test_proj,test_labels)

W = LDA(data,labels,25)
train_proj=train@W
test_proj=test@W
recognition(train_proj, train_labels, test_proj,test_labels)

#part3
print('part3:')

kernel=linear_Kernel(data) 
#kernel=polynomial_Kernel(data,0.000001,0,3)
#kernel=RBF_Kernel(data,0.000000001)

data_proj=kernel_PCA(kernel,25)
train_proj=data_proj[:train.shape[0],:]
test_proj=data_proj[train.shape[0]:,:]
recognition(train_proj, train_labels, test_proj,test_labels)

data_proj=kernel_LDA(kernel,labels,25)
train_proj=data_proj[:train.shape[0],:]
test_proj=data_proj[train.shape[0]:,:]
recognition(train_proj,train_labels,test_proj,test_labels)
