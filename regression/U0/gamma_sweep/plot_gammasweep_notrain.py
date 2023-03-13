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
alpha0 = 0
temp_gamma = []
temp_errors = []
#temp_alphas = []
for line in filelines:
    try:
        if line.split()[0] == 'alpha=' and float(line.split()[1]) > alpha0:
            alphas.append(float(line.split()[1]))
            alpha0 = float(line.split()[1])
            gammas.append(temp_gamma)
            temp_gamma = []
            errors.append(temp_errors)
            temp_errors = []
        if line.split()[0] == 'gamma=':
            temp_gamma.append(float(line.split()[1]))
        if line.split()[0] == 'error:':
            temp_errors.append(float(line.split()[1]))
    except IndexError:
        pass

gammas.append(temp_gamma)
temp_gamma = []
errors.append(temp_errors)
temp_errors = []
gammas.pop(0)
errors.pop(0)
print(len(alphas))
#print(alphas)
print(len(gammas))
#print(gammas)
#print(gammas[1])
print(len(errors))
#print(errors)
#print(len(errors[1]))


plt.figure()
for i,alpha in enumerate(alphas[0:10]):
    plt.plot(gammas[i][1:],errors[i][1:],'.',label=str(alpha))
    min_err = min(errors[i])
    min_ind = errors[i].index(min_err)
    print(alpha,gammas[i][min_ind],min_err)
plt.legend()
plt.show()
