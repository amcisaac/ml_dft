import numpy as np
from sklearn.kernel_ridge import KernelRidge
import sys


def xval_learning_alg(data, Es, k,a):
    d,n=data.shape
    print('Input data shape:',data.shape,flush=True)
    print('Energy data shape:',Es.shape,flush=True)
    #print(d,n)
    #print(k)
    #print(Es[0:5])
    split_data=np.array_split(data,k,axis=0) # transposed from what we had in class
    split_labels = np.array_split(Es,k,axis=0)
    # print(len(split_data))
    # print(len(split_labels))
    #print(split_data[0].shape)
    #print(split_labels[0].shape)

    print('Training data size:',data.shape[0] - split_data[0].shape[0],flush=True)
    print('Testing data size:',split_data[0].shape[0],flush=True)
    print('',flush=True)
    # print(Es)
    # print(split_labels)
    avg_test_err=0
    avg_train_err = 0
    for j in range(0,k):
        print('Cross-validation round ', j,flush=True)
        # process data
        if j != 0 and j != k-1:
            test1 = np.concatenate(split_data[0:j],axis=0)
            test2 = np.concatenate(split_data[j+1:],axis=0)
            train_data = np.concatenate((test1,test2),axis=0)

            test1l = np.concatenate(split_labels[0:j],axis=0)
            test2l = np.concatenate(split_labels[j+1:],axis=0)
            train_Es = np.concatenate((test1l,test2l),axis=0)

        elif j == 0:
            train_data = np.concatenate(split_data[j+1:],axis=0)
            train_Es = np.concatenate(split_labels[j+1:],axis=0)


        elif j == k-1:
            #print('end')
            train_data = np.concatenate(split_data[0:j],axis=0)
            train_Es = np.concatenate(split_labels[0:j],axis=0)
        test_data = split_data[j]
        test_Es = split_labels[j]

        ridge = KernelRidge(kernel='polynomial',degree=2,alpha=a)
        kr = ridge.fit(train_data,train_Es)
        y_kr = kr.predict(test_data)
        score = kr.score(test_data,test_Es)
        print('first five predictions:',y_kr[0:5],flush=True)
        print('first five energies:   ',test_Es[0:5],flush=True)
        print('score:', score,flush=True)

        err = np.mean(np.abs(y_kr - test_Es))
        print('test error:', err,flush=True)

        y_kr_train = kr.predict(train_data)
        err_train = np.mean(np.abs(y_kr_train - train_Es))
        print('train error:', err_train,flush=True)
        print('',flush=True)

        avg_test_err += err
        avg_train_err += err_train
    return avg_test_err/k, avg_train_err/k

# COULOMB MATRICES
inp_data = np.load(sys.argv[1]) # input data should be sys.argv 1, in the form of a .npy file
# ENERGY DATA
U0,U298,H298,G298,Cv298 = np.loadtxt(sys.argv[2],delimiter=',',skiprows=1,usecols=(11,12,13,14,15),unpack=True,comments=None) # prediction data should be in sys.argv 2 in csv file

# SPLIT FOR TESTING VS TRAINING, indicate what E to use
#split = int(U0.shape[0] * 0.2)
energy_type = 'U0'
train_set = U0
test_set = U0
print('Using data for ',energy_type,flush=True)


print('Kernel type: Gaussian (RBF)',flush=True)

a = 1.5e-5
print('alpha=',a,flush=True)
print('',flush=True)
split_list = np.arange(0.1,0.6,0.1)
#split_list = np.arange(0.6,0.8,0.1)
#split_list = np.arange(0.8,1.0,0.1)
for split in split_list:
    i = int(U0.shape[0] * split)
    data = inp_data[:i,:]
    Es = train_set[:i]
    print("Split index:",i)
    print('Total input data set size:',data.shape,flush=True)
    print("Total Es data set size:",Es.shape,flush=True)

    avg_test_err,avg_train_err = xval_learning_alg(data, Es,5,a)

    print("Cross-validation training error: ",avg_train_err,flush=True)
    print("Cross-validation testing error: ",avg_test_err,flush=True)
    print('',flush=True)
