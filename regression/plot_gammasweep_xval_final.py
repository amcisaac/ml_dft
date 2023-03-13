import numpy as np
from matplotlib import pyplot as plt
import sys

inpfile = sys.argv[1]
f = open(inpfile,'r')
filelines = f.readlines()
f.close()

alphas = []
gammas = []
errors = []
train_errors = []
alpha0 = -1
temp_gamma = []
temp_errors = []
temp_train_errors = []
#temp_alphas = []
for line in filelines:
    try:
        if line.split()[0] == 'alpha=' and float(line.split()[1]) != alpha0:
            alphas.append(float(line.split()[1]))
            alpha0 = float(line.split()[1])
            gammas.append(temp_gamma)
            temp_gamma = []
            errors.append(temp_errors)
            temp_errors = []
            train_errors.append(temp_train_errors)
            temp_train_errors = []
        if line.split()[0] == 'gamma=':
            temp_gamma.append(float(line.split()[1]))
        if line.split()[1] == 'testing':
            #print(line.split())
            temp_errors.append(float(line.split()[-1]))
        if line.split()[1] == 'training':
            #print(line.split())
            temp_train_errors.append(float(line.split()[-1]))
    except IndexError:
        pass

gammas.append(temp_gamma)
temp_gamma = []
errors.append(temp_errors)
temp_errors = []
train_errors.append(temp_train_errors)
temp_train_errors = []
gammas.pop(0)
errors.pop(0)
train_errors.pop(0)
# print(len(alphas))
# #print(alphas)
# print(len(gammas))
#print(errors[0])
# #print(gammas[1])
# print(len(errors))
# print(len(train_errors))
#print(errors)
#print(len(errors[1]))

errors_arr = np.array([np.array(x) for x in errors])
#print(errors)
#print(errors.type)
errors_arr = errors_arr*627.509

min_errs_kcalmol=[]
min_alphas = []

print('Testing error:')

plt.figure()
for i,alpha in enumerate(alphas[0:4]):

    plt.plot(gammas[i],errors_arr[i],'.',label=r'$\alpha=$'+str(alpha))
    min_err = min(errors[i])
    min_ind = errors[i].index(min_err)
    print(alpha,gammas[i][min_ind],min_err*627.509)
    min_errs_kcalmol.append(min_err*627.509)
    min_alphas.append(alpha)
for i,alpha in enumerate(alphas[-2:]):
    i += len(alphas)-2
    plt.plot(gammas[i],errors_arr[i],'.',label=r'$\alpha=$'+str(alpha))
    min_err = min(errors[i])
    min_ind = errors[i].index(min_err)
    print(alpha,gammas[i][min_ind],min_err*627.509)
    min_errs_kcalmol.append(min_err*627.509)
    min_alphas.append(alpha)
plt.xlim(-1e-6,0.000101)
plt.ylim(35,46)
#plt.title('Gaussian Sweep Over gamma')
plt.legend(loc=1)
plt.xlabel(r'$\gamma$')
plt.ylabel('Testing error (kcal/mol)')
#plt.legend()

#plt.show()
# print('Training error:')
# plt.figure()
# for i,alpha in enumerate(alphas[5:10]):
#     plt.plot(gammas[i][1:],train_errors[i][1:],'.',label=str(alpha))
#     min_train_err = min(train_errors[i])
#     min_train_ind = train_errors[i].index(min_train_err)
#     print(alpha,gammas[i][min_train_ind],min_train_err)
# plt.legend()
# plt.title("train")
plt.show()

plt.figure()
plt.plot(min_alphas,min_errs_kcalmol,'.-')
plt.xlabel(r'$\alpha$')
plt.ylabel('Minimum testing error (kcal/mol)')
#plt.xlim(-1e-11,1.1e-9)
#plt.ylim(35.64,35.78)
plt.show()

plt.figure()
plt.plot(min_alphas,min_errs_kcalmol,'.-')
plt.xlabel(r'$\alpha$')
plt.ylabel('Minimum testing error (kcal/mol)')
plt.xlim(-1e-11,1.1e-9)
plt.ylim(35.64,35.78)
plt.show()
