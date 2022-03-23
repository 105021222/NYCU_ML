import numpy as np

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
