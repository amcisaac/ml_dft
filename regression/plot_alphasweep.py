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
for line in filelines:
    try:
        if line.split()[0] == 'alpha=':
            alphas.append(float(line.split()[1]))

        if line.split()[0] == 'error:':
            errors.append(float(line.split()[1]))
    except IndexError:
        pass

# gammas.append(temp_gamma)
# temp_gamma = []
# errors.append(temp_errors)
# temp_errors = []
# gammas.pop(0)
# errors.pop(0)
# print(len(alphas))
# print(alphas)
# print(len(gammas))
# #print(gammas)
# print(gammas[1])
# print(len(errors))
# #print(errors)
# print(len(errors[1]))


plt.figure()
#for i,alpha in enumerate(alphas):
plt.plot(alphas[2:],errors[2:],'.')
min_err = min(errors)*627.509
print(min_err)
#min_ind = errors[i].index(min_err)
#print(alpha,gammas[[min_ind],min_err)
plt.legend()
plt.savefig(inpfile+".pdf")
