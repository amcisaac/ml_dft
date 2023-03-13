import numpy as np
from sklearn.kernel_ridge import KernelRidge
import sys

# COULOMB MATRICES
inp_data = np.load(sys.argv[1]) # input data should be sys.argv 1, in the form of a .npy file
inp_data_test = inp_data[:,[0]] # just first eigenvalue
# ENERGY DATA
U0,U298,H298,G298,Cv298 = np.loadtxt(sys.argv[2],delimiter=',',skiprows=1,usecols=(11,12,13,14,15),unpack=True,comments=None) # prediction data should be in sys.argv 2 in csv file

# SPLIT FOR TESTING VS TRAINING, indicate what E to use
split = int(U0.shape[0] * 0.05)
energy_type = 'U0'
train_set = U0
test_set = U0
print('Using data for ',energy_type,flush=True)

# TRAINING DATA
train_M = inp_data[:split,:]
train_M_test = train_M[:,[0]]
train_E = train_set[:split]
print('Training data size: ',train_E.shape[0],flush=True)

# TESTING DATA
test_M = inp_data[split:,:]
test_M_test = test_M[:,[0]]
test_E = test_set[split:]
print('Testing data size: ',test_E.shape[0],flush=True)


print('Kernel type: Laplacian',flush=True)
#BEST alpha= 0.273 error=0.189791021959
# alpha_list = [1e-4]#np.arange(1e-8,0.02,0.001)
# gamma_list = [1e-5]#np.arange(1e-8,0.00201,0.00005) # gamma = 1/(A**2)
# print(len(gamma_list)*len(alpha_list))

alpha_list = np.arange(1e-12,1.1e-8,5e-10)
#gamma_list = np.arange(1e-8,0.2500001,0.0025)
gamma_list = np.arange(1e-12,1e-5,5e-8)
for a in alpha_list:
    # if a == 1e-8:
    #     gamma_list = np.arange(1e-8,0.002,0.00001)
    # if a == 0.1:
    #     gamma_list = np.arange(1e-8,0.004,0.00001)
    # else:
    #     gamma_list = np.arange(1e-8,0.003,0.00001)
    for g in gamma_list:
        print('alpha=',a,flush=True)
        print('gamma=',g,flush=True)
        ridge = KernelRidge(kernel='laplacian',gamma=g,degree=3,alpha=a)
        kr = ridge.fit(train_M,train_E)
        y_kr = kr.predict(test_M)
        score = kr.score(test_M,test_E)
        print('first five predictions:',y_kr[0:5],flush=True)
        print('first five energies:   ',test_E[0:5],flush=True)
        print('score:', score,flush=True)

        err = np.mean(np.abs(y_kr - test_E))
        print('test error:', err,flush=True)

        y_kr_train = kr.predict(train_M)
        err_train = np.mean(np.abs(y_kr_train - train_E))
        print('train error:', err_train,flush=True)
        print('',flush=True)
