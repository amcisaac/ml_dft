import numpy as np
from sklearn.kernel_ridge import KernelRidge
import sys

dft,pred,sigerr,abserr = np.loadtxt(sys.argv[1],delimiter=',',skiprows=1,usecols=(1,2,3,4),unpack=True,comments=None) # prediction data should be in sys.argv 2 in csv file
smiles = np.loadtxt(sys.argv[1],delimiter=',',skiprows=1,usecols=(0),unpack=True,comments=None,dtype=str)

max_errs = abserr[abserr>=0.2]
max_signederr = sigerr[abserr>=0.2]
max_smiles = smiles[abserr>=0.2]

f1 = open(sys.argv[2],'w')
f1.write('SMILES,Signed error,Absolute error')
for i in range(0,len(max_errs)):
    f1.write(max_smiles[i]+','+str(max_signederr[i])+','+str(max_errs[i])+'\n')
f1.close()

min_errs = abserr[abserr<=0.001]
min_smiles = smiles[abserr<=0.001]
min_signederr = sigerr[abserr<=0.001]

f2 = open(sys.argv[3],'w')
f2.write('SMILES,Signed error,Absolute error')
for i in range(0,len(min_errs)):
    f2.write(min_smiles[i]+','+str(min_signederr[i])+','+str(min_errs[i])+'\n')
f2.close()
