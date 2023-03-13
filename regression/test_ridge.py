import numpy as np
from sklearn.kernel_ridge import KernelRidge
import sys

inp_data = np.load(sys.argv[1]) # input data should be sys.argv 1, in the form of a .npy file
inp_data_test = inp_data[:,[0]] # just first eigenvalue
#print(inp_data[1])
#print(inp_data_test[1])
#print(inp_data[0])
U_0 = np.loadtxt(sys.argv[2],delimiter=',',skiprows=1,usecols=(11,),unpack=True,comments=None) # prediction data should be in sys.argv 2 in csv file

split = int(U_0.shape[0] * 0.05)

train_data = inp_data[:split,:]
train_data_test = train_data[:,[0]]
train_U = U_0[:split]
print(train_data.shape,train_U.shape)

test_data = inp_data[split:,:]
test_data_test = test_data[:,[0]]
test_U = U_0[split:]
print(test_data.shape,test_U.shape)

print(inp_data.shape,U_0.shape)

#print(U_0[0])
ridge = KernelRidge(kernel='linear',degree=1,alpha=1e-5)
kr = ridge.fit(train_data,train_U)
y_kr = kr.predict(test_data)
score = kr.score(test_data,test_U)
print(y_kr[0:5])
print(test_U[0:5])
print(score)

err = np.mean(np.abs(y_kr - test_U))
print(err)
