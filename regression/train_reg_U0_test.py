import numpy as np
from sklearn.kernel_ridge import KernelRidge
import sys
from sklearn.metrics.pairwise import rbf_kernel as kernel

# COULOMB MATRICES
inp_data = np.load(sys.argv[1]) # input data should be sys.argv 1, in the form of a .npy file
inp_data_test = inp_data[:,[0]] # just first eigenvalue
# ENERGY DATA
U0,U298,H298,G298,Cv298 = np.loadtxt(sys.argv[2],delimiter=',',skiprows=1,usecols=(11,12,13,14,15),unpack=True,comments=None) # prediction data should be in sys.argv 2 in csv file
smiles = np.loadtxt(sys.argv[2],delimiter=',',skiprows=1,usecols=(0),unpack=True,comments=None,dtype=str)
#inp_data_mat = np.load(sys.argv[3])
#print(inp_data_mat.shape)

# SPLIT FOR TESTING VS TRAINING, indicate what E to use
split = 20000 #int(U0.shape[0] * 0.2) # 0.2*133000 ~ 30000
energy_type = 'U0'
train_set = U0
test_set = U0
print('Using data for ',energy_type,flush=True)

# # TRAINING DATA
train_data = inp_data[:split,:]
#train_data_mat = inp_data_mat[:split,:,:]
#print(train_data_mat.shape)
print(train_data.shape)
train_Es = train_set[:split]
print('Training set size:',len(train_Es),flush=True)

test_data = inp_data[split:,:]
test_Es = test_set[split:]
print('Testing set size:',len(test_Es),flush=True)

a = 1e-11
g = 5e-6
print('Using Gaussian (RBF) kernel with gamma =',g,'alpha=',a,flush=True)
print('variance:', np.sqrt(1/g))

K = kernel(train_data,gamma=g)
mol_n = 1
#argmax = np.argmax(K[mol_n][1:])
#print(argmax)
#print(K[0][argmax+1])
#argsort = np.flip(np.argsort(K[mol_n]))
argsort = np.argsort(K[mol_n])

#print(argsort[:5]+2)
n_sim = 10
print(argsort[:n_sim]+2)
print(K[mol_n][argsort[:n_sim]])
sim_mols = train_data[argsort[:n_sim]]
sim_smiles = smiles[argsort[:n_sim]]
#sim_mats = train_data_mat[argsort[:n_sim],:,:]
sim_Es = train_Es[argsort[:n_sim]]
#print(train_data[0])
for i,mol in enumerate(sim_mols):
    print(sim_smiles[i],sim_Es[i])
#    print(mol, np.abs(train_data[0]-mol))
#    if i == 0:
#        print(sim_mats[i,:17,:17])
#    if i == 1:
#        print(sim_mats[i,:21,:21])


# ridge = KernelRidge(kernel='rbf',gamma=g,alpha=a)
# kr = ridge.fit(train_data,train_Es)
# params= kr.dual_coef_
# print("Params ", params)
# for param in params:
#     print(param)
