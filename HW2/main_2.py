from math import factorial


def Likelihood(p,a,b):
    return (factorial(a+b)/(factorial(a)*factorial(b)))*(p**a)*((1-p)**b)


a=int(input("Input initial a:"))
b=int(input("Input initial b:"))
fp=open("testfile.txt","r")
line=fp.readline()
i=1
while line:
    ones=line.count("1")
    zeros=line.count("0")
    print("case",i,":",line,end='')
    p=ones/(ones+zeros)
    print("Likelihood:",Likelihood(p,ones,zeros))
    print("Beta prior:     a={} b={}".format(a,b))
    a+=ones
    b+=zeros
    print("Beta posterior: a={} b={}\n".format(a,b))
    i+=1
    line=fp.readline()

fp.close()
