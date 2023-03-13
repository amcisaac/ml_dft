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


# print('Kernel type: polynomial order 1',flush=True)
# #BEST alpha= 0.273 error=0.189791021959
# alpha_list = np.arange(0.2,0.3,0.0025)
# for a in alpha_list:
#    print('alpha=',a,flush=True)
#    #ridge = KernelRidge(kernel='linear',gamma=.001,degree=3,alpha=a)
#    ridge = KernelRidge(kernel='linear',degree=3,alpha=a)
#    kr = ridge.fit(train_M,train_E)
#    y_kr = kr.predict(test_M)
#    score = kr.score(test_M,test_E)
#    print('first five predictions:',y_kr[0:5],flush=True)
#    print('first five energies:   ',test_E[0:5],flush=True)
#    print('score:', score,flush=True)
#
#    err = np.mean(np.abs(y_kr - test_E))
#    print('error:', err,flush=True)
#    print('',flush=True)

# print('Kernel type: polynomial order 2',flush=True)
# #BEST alpha= 1e-05 error=0.0690799269975
# alpha_list = np.arange(0.0005,0.002,0.00005)
# for a in alpha_list:
#     print('alpha=',a,flush=True)
#     #ridge = KernelRidge(kernel='polynomial',gamma=.001,degree=2,alpha=a)
#     ridge = KernelRidge(kernel='polynomial',degree=2,alpha=a)
#     kr = ridge.fit(train_M,train_E)
#     y_kr = kr.predict(test_M)
#     score = kr.score(test_M,test_E)
#     print('first five predictions:',y_kr[0:5],flush=True)
#     print('first five energies:   ',test_E[0:5],flush=True)
#     print('score:', score,flush=True)
#
#     err = np.mean(np.abs(y_kr - test_E))
#     print('error:', err,flush=True)
#     print('',flush=True)

print('Kernel type: polynomial order 3',flush=True)
#BEST alpha= 0.0005 error=0.0639994416418
alpha_list = np.arange(0.01,0.05001,0.002)
for a in alpha_list:
   print('alpha=',a,flush=True)
   #ridge = KernelRidge(kernel='polynomial',gamma=.001,degree=3,alpha=a)
   ridge = KernelRidge(kernel='polynomial',degree=3,alpha=a)
   kr = ridge.fit(train_M,train_E)
   y_kr = kr.predict(test_M)
   score = kr.score(test_M,test_E)
   print('first five predictions:',y_kr[0:5],flush=True)
   print('first five energies:   ',test_E[0:5],flush=True)
   print('score:', score,flush=True)

   err = np.mean(np.abs(y_kr - test_E))
   print('error:', err,flush=True)
   print('',flush=True)

# print('Kernel type: polynomial order 4')
# #BEST alpha= 0.0025 error=0.067240898017
# alpha_list = np.arange(0.001,0.003,0.0005)
# for a in alpha_list:
#    print('alpha=',a)
#    ridge = KernelRidge(kernel='polynomial',gamma=.001,degree=4,alpha=a)
#    kr = ridge.fit(train_M,train_E)
#    y_kr = kr.predict(test_M)
#    score = kr.score(test_M,test_E)
#    print('first five predictions:',y_kr[0:5])
#    print('first five energies:   ',test_E[0:5])
#    print('score:', score)
#
#    err = np.mean(np.abs(y_kr - test_E))
#    print('error:', err)
#    print('')

# print('Kernel type: polynomial order 5')
# #BEST alpha= 0.124 error=0.0810995525409
# alpha_list = np.arange(0.1,0.15,0.001)
# for a in alpha_list:
#    print('alpha=',a)
#    ridge = KernelRidge(kernel='polynomial',gamma=.001,degree=5,alpha=a)
#    kr = ridge.fit(train_M,train_E)
#    y_kr = kr.predict(test_M)
#    score = kr.score(test_M,test_E)
#    print('first five predictions:',y_kr[0:5])
#    print('first five energies:   ',test_E[0:5])
#    print('score:', score)
#
#    err = np.mean(np.abs(y_kr - test_E))
#    print('error:', err)
#    print('')
