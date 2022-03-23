import numpy as np
import matplotlib.pyplot as plt
import gzip
import math
from discrete import test_discrete
from continuous import test_continuous

def read_images(filename):
    fp=gzip.open(filename,'r')
    magic_number=int.from_bytes(fp.read(4),byteorder='big')
    numbers=int.from_bytes(fp.read(4),byteorder='big')
    rows=int.from_bytes(fp.read(4),byteorder='big')
    columns=int.from_bytes(fp.read(4),byteorder='big')
    
    buf=fp.read(rows*columns*numbers)
    images=np.frombuffer(buf,dtype=np.uint8)
    images=images.reshape(numbers,rows,columns)

    return images


def read_labels(filename):
    fp=gzip.open(filename,'r')
    magic_number=int.from_bytes(fp.read(4),byteorder='big')
    numbers=int.from_bytes(fp.read(4),byteorder='big')
    
    buf=fp.read(numbers)
    labels=np.frombuffer(buf,dtype=np.uint8)
    
    return labels



images=read_images('train-images-idx3-ubyte.gz')
labels=read_labels('train-labels-idx1-ubyte.gz')
test_images=read_images('t10k-images-idx3-ubyte.gz')
test_labels=read_labels('t10k-labels-idx1-ubyte.gz')

toggle=input('Toggle option ( 0:discrete mode / 1:continuous mode ) :')
if toggle=='0':
    test_discrete(images,labels,test_images,test_labels)
elif toggle=='1':
    test_continuous(images,labels,test_images,test_labels)


'''
for i in range(1):
    tmp=images[i,:,:]
    plt.imshow(tmp)
    plt.show()
'''




