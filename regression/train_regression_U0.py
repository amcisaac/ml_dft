import numpy as np
from sklearn.kernel_ridge import KernelRidge
import sys

# COULOMB MATRICES
inp_data = np.load(sys.argv[1]) # input data should be sys.argv 1, in the form of a .npy file
inp_data_test = inp_data[:,[0]] # just first eigenvalue
# ENERGY DATA
U0,U298,H298,G298,Cv298 = np.loadtxt(sys.argv[2],delimiter=',',skiprows=1,usecols=(11,12,13,14,15),unpack=True,comments=None) # prediction data should be in sys.argv 2 in csv file

# SPLIT FOR TESTING VS TRAINING, indicate what E to use
split = int(U0.shape[0] * 0.2) # 0.2*133000 ~ 30000
energy_type = 'U0'
train_set = U0
test_set = U0
print('Using data for ',energy_type,flush=True)

# # TRAINING DATA
train_data = inp_data[:split,:]
train_Es = train_set[:split]
print('Training set size:',len(train_Es),flush=True)

test_data = inp_data[split:,:]
test_Es = test_set[split:]
print('Testing set size:',len(test_Es),flush=True)

a = 1e-11
g = 5e-6
print('Using Gaussian (RBF) kernel with gamma =',g,'alpha=',a,flush=True)

ridge = KernelRidge(kernel='rbf',gamma=g,alpha=a)
kr = ridge.fit(train_data,train_Es)

y_kr_train = kr.predict(train_data)
err_train = np.mean(np.abs(y_kr_train - train_Es))
print('Training error:', err_train,flush=True)
print('',flush=True)

y_kr = kr.predict(test_data)
score = kr.score(test_data,test_Es)
print('Testing score:', score,flush=True)
err = np.mean(np.abs(y_kr - test_Es))
print('Testing error:', err,flush=True)

output_file_train = sys.argv[3]
output_file_test = sys.argv[4]
f1 = open(output_file_train,'w')
f1.write('U0 (DFT),U0 (predicted),Signed error,Absolute Error\n')
for i in range(0,len(y_kr_train)):
    f1.write(str(train_Es[i])+','+str(y_kr_train[i])+','+str(y_kr_train[i]-train_Es[i])+','+str(np.abs(y_kr_train[i]-train_Es[i]))+'\n')
f1.close()

f2 = open(output_file_test,'w')
f2.write('U0 (DFT),U0 (predicted),Signed error,Absolute Error')
for i in range(0,len(y_kr)):
    f2.write(str(test_Es[i])+','+str(y_kr[i])+','+str(y_kr[i]-test_Es[i])+','+str(np.abs(y_kr[i]-test_Es[i]))+'\n')
f2.close()

# print('first five training predictions:',y_kr_train[0:5],flush=True)
# print('first five energies:   ',train_Es[0:5],flush=True)
#
# print('first five testing predictions:',y_kr[0:5],flush=True)
# print('first five energies:   ',test_Es[0:5],flush=True)
