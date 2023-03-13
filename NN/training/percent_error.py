import numpy as np
from sklearn.kernel_ridge import KernelRidge
import sys

actual,err = np.loadtxt(sys.argv[1],delimiter=',',skiprows=1,usecols=(1,4),unpack=True,comments=None)

perc_err = 100*np.abs(err)/actual
print(np.average(perc_err))
print(np.min(perc_err))
