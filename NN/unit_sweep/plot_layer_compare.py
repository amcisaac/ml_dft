import numpy as np
import matplotlib.pyplot as plt

loss = np.array([0.0887429819139838,0.046937022883296015,0.054959208297133445,0.03527873844325542,0.10529983037233352])*627.509
N_layer = np.array(range(2,7))
MAE = np.array([0.24514019958496092,0.17217196075439453,0.1822638787841797,0.1394982815551758,0.26365826232910156])*627.509


plt.figure()
#plt.plot(N_layer,loss,'.-',label='Testing')
plt.plot(N_layer-1,MAE,'.-',label='Testing')
#plt.legend(loc=2)
plt.ylim(60,.3*627.509)
plt.ylabel('Testing Error (kcal/mol)')
plt.xlabel('Number of layers')
plt.savefig('N_layer_vs_xval_MAE.pdf')
