import numpy as np
import matplotlib.pyplot as plt
import sys

train_size,err = np.loadtxt(sys.argv[1],delimiter=',',skiprows=1,usecols=(0,1),unpack=True,comments=None)

train_size2,err2 = np.loadtxt(sys.argv[2],delimiter=',',skiprows=1,usecols=(0,1),unpack=True,comments=None)

plt.figure()
plt.plot(train_size,err*627.509,'.-',label='Gaussian')
plt.plot(train_size2,err2*627.509,'.-',label='Quadratic')
plt.xlabel('Training set size')
plt.ylabel('Testing error (kcal/mol)')
plt.legend()
plt.show()
