import numpy as np
from sklearn.kernel_ridge import KernelRidge
import sys

U0,U298,H298,G298,Cv298 = np.loadtxt(sys.argv[1],delimiter=',',skiprows=1,usecols=(11,12,13,14,15),unpack=True,comments=None) # prediction data should be in sys.argv 2 in csv file


print('U298 MAE:',np.mean(np.abs(U0-U298))*627.509)
print('H298 MAE:',np.mean(np.abs(U0-H298))*627.509)
print('G298 MAE:',np.mean(np.abs(U0-G298))*627.509)
