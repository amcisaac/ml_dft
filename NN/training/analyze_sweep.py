import numpy as np
import matplotlib.pyplot as plt
import sys

inpfile = sys.argv[1]
f = open(inpfile,'r')
filelines = f.readlines()
f.close()

y = int(sys.argv[2]) # number of layers

N_units = []
epochs = range(0,20)
trainloss = []
trainmae = []
trainmse = []
testloss = []
testmae = []
testmse = []

N_units_int = []

N = 0
for line in filelines:
    if len(line.split()) > 2 and line.split()[2] == 'units':
        unit_list = [x.strip('[],') for x in line.split()[-y:]]
        unit_list_int = [int(x.strip('[],')) for x in line.split()[-y:]]
        N_units.append(unit_list)
        N_units_int.append(unit_list_int)
        if N != 0:
            trainloss.append(temp_trainloss)
            trainmae.append(temp_trainmae)
            trainmse.append(temp_trainmse)
            testloss.append(temp_testloss)
            testmae.append(temp_testmae)
            testmse.append(temp_testmse)
        temp_trainloss = []
        temp_trainmae = []
        temp_trainmse = []
        temp_testloss = []
        temp_testmae = []
        temp_testmse = []
        N += 1
        #new_flag = False
    if len(line.split())>0 and line.split()[0] == 'Training':
        if line.split()[1] == 'loss:':
            temp_trainloss.append(float(line.split()[2]))
        elif line.split()[1] == 'MAE:':
            temp_trainmae.append(float(line.split()[2]))
        elif line.split()[1] == 'MSE:':
            temp_trainmse.append(float(line.split()[2]))
    if len(line.split())>0 and line.split()[0] == 'Testing':
        if line.split()[1] == 'loss:':
            temp_testloss.append(float(line.split()[2]))
        elif line.split()[1] == 'MAE:':
            temp_testmae.append(float(line.split()[2]))
        elif line.split()[1] == 'MSE:':
            temp_testmse.append(float(line.split()[2]))

trainloss.append(temp_trainloss)
trainmae.append(temp_trainmae)
trainmse.append(temp_trainmse)
testloss.append(temp_testloss)
testmae.append(temp_testmae)
testmse.append(temp_testmse)

#print(' '.join(N_units[0]))
lowest_test_loss = []
lowest_test_mae = []
unit_str = []

final_test_loss = []
final_test_mae = []
final_train_loss = []
final_train_mae = []
for i,x in enumerate(trainloss):
        train_loss_i = trainloss[i][:-1]
        train_mae_i = trainmae[i][:-1]
        train_mse_i = trainmse[i][:-1]
        test_loss_i = testloss[i][:-1]
        test_mae_i = testmae[i][:-1]
        test_mse_i = testmse[i][:-1]

        final_test_loss.append(test_loss_i[-1])
        final_test_mae.append(test_mae_i[-1])
        final_train_loss.append(train_loss_i[-1])
        final_train_mae.append(train_mae_i[-1])

        unit_str.append(' '.join(N_units[i]))

        plt.figure()
        plt.plot(epochs,np.array(train_loss_i)*627.509,label = 'Training loss')
        plt.plot(epochs,np.array(test_loss_i)*627.509,label = 'Testing loss')
        plt.ylim(0,627.509)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss (kcal/mol)')
        #plt.title('Number of units: '+' '.join(N_units[i]))
        #plt.savefig('err_vs_epoch_5layer_35/err_vs_epoch_'+'_'.join(N_units[i])+'.pdf')
        plt.show()

        train_mae_f = trainmae[i][-1]
        train_mse_f = trainmse[i][-1]
        test_loss_f = testloss[i][-1]
        test_mae_f = testmae[i][-1]
        test_mse_f = testmse[i][-1]

        lowest_test_loss.append(min(test_loss_i))
        lowest_test_mae.append(min(test_mae_i))
