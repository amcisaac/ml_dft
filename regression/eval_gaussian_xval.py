import numpy as np
from sklearn.kernel_ridge import KernelRidge
import sys

# COULOMB MATRICES
inp_data = np.load(sys.argv[1]) # input data should be sys.argv 1, in the form of a .npy file
inp_data_test = inp_data[:,[0]] # just first eigenvalue
# ENERGY DATA
U0,U298,H298,G298,Cv298 = np.loadtxt(sys.argv[2],delimiter=',',skiprows=1,usecols=(11,12,13,14,15),unpack=True,comments=None) # prediction data should be in sys.argv 2 in csv file

# SPLIT FOR TESTING VS TRAINING, indicate what E to use
split = int(U0.shape[0] * 0.2)
energy_type = 'U0'
train_set = U0
test_set = U0
print('Using data for ',energy_type,flush=True)

# # TRAINING DATA
data = inp_data[:split,:]
# train_M_test = train_M[:,[0]]
Es = train_set[:split]
# print('Training data size: ',train_E.shape[0],flush=True)
#
# # TESTING DATA
# test_M = inp_data[split:,:]
# test_M_test = test_M[:,[0]]
# test_E = test_set[split:]
# print('Testing data size: ',test_E.shape[0],flush=True)


def xval_learning_alg(data, Es, k,kernel_name,g,a):
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

        ridge = KernelRidge(kernel=kernel_name,gamma=g,alpha=a)
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
        #score_j=eval_classifier(learner,D_minus_j,lab_min_j,split_data[j],split_labels[j])
        #avg_score = avg_score + score_j
    return avg_test_err/k, avg_train_err/k



print('Kernel type: Gaussian (RBF)',flush=True)
#BEST alpha= 0.273 error=0.189791021959
alpha_list = np.array([1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3])#[1e-5]#np.arange(1e-12,1.1e-8,5e-10)
#gamma_list = np.arange(1e-8,0.2500001,0.0025)
#gamma_list =[1e-5]
#print(len(gamma_list)*len(alpha_list))
for a in alpha_list:
    if a == 1e-3:
        gamma_list = np.concatenate((np.arange(5e-5,1e-4,1e-5),np.arange(1e-4,5e-4,5e-5))) #np.arange(5e-6,1e-3,5e-5) # 20
    if a == 1e-4:
        gamma_list = np.concatenate((np.arange(4e-5,1.501e-4,5e-6),np.array([2e-4,2.5e-4,3e-4]))) #np.concatenate((np.arange(2e-6,5.2e-5,2e-6),np.array([7.5e-5,1.5e-4,3e-4]))) # 28
    # if a == 1e-5 or a == 1e-6 or a == 1e-7 or a == 1e-8:
    #     gamma_list = np.concatenate((np.concatenate((np.arange(2e-6,2.2e-5,2e-6),np.arange(2.5e-5,1.05e-4,5e-6))),np.array([3e-4,8e-4]))) # 28
    if a == 1e-6: #or a == 1e-7: #or a == 1e-8:
        gamma_list = np.concatenate((np.concatenate((np.arange(2e-6,2.2e-5,5e-6),np.arange(2.5e-5,1.55e-4,1e-5))),np.array([2.5e-4,3.5e-4]))) # 28
    if a == 1e-5:
        gamma_list = np.concatenate((np.concatenate((np.arange(2e-6,2.2e-5,5e-6),np.arange(2.5e-5,2.55e-4,1e-5))),np.array([3e-4,3.5e-4])))
    if a == 1e-9:
        gamma_list = np.concatenate((np.arange(1e-6,2.1e-5,1e-6),np.array([2e-7,5e-7,3e-5,4e-5,5e-5]))) # 14
    if a == 1e-10:
        gamma_list = np.concatenate((np.arange(2e-7,1.1e-5,5e-7),np.arange(1.1e-5,5.1e-5,5e-6))) # 26
    if a == 1e-11:
        gamma_list = np.concatenate((np.arange(2e-7,2.2e-6,2e-7),np.arange(3e-6,2e-5,1e-6))) # 15
    if a == 1e-12:
        gamma_list = np.concatenate((np.arange(1e-7,1.1e-6,1e-7),np.arange(1.1e-6,1e-5,1e-6))) # 13
    for g in gamma_list:
        print('alpha=',a,flush=True)
        print('gamma=',g,flush=True)

        avg_test_err,avg_train_err = xval_learning_alg(data, Es, 5,'rbf',g,a)

        print("Cross-validation training error: ",avg_train_err,flush=True)
        print("Cross-validation testing error: ",avg_test_err,flush=True)
        print('',flush=True)
