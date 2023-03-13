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
training_size = []

N_units_int = []

N = 0
for line in filelines:
    #print(line)
    if len(line.split()) > 3 and line.split()[0] == 'Training' and line.split()[2] =='size:':
        #unit_list = [x.strip('[],') for x in line.split()[-y:]]
        #unit_list_int = [int(x.strip('[],')) for x in line.split()[-y:]]
        #N_units.append(unit_list)
        #N_units_int.append(unit_list_int)
        training_size.append(float(line.split()[3]))
        print('here')
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
    elif len(line.split())>0 and line.split()[0] == 'Training':
        if line.split()[1] == 'loss:':
            temp_trainloss.append(float(line.split()[2]))
        elif line.split()[1] == 'MAE:':
            temp_trainmae.append(float(line.split()[2]))
        elif line.split()[1] == 'MSE:':
            temp_trainmse.append(float(line.split()[2]))
    if len(line.split())>0 and len(line.split())<4 and line.split()[0] == 'Testing':
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

        #unit_str.append(' '.join(N_units[i]))

        # plt.figure()
        # plt.plot(epochs,train_loss_i,label = 'training loss')
        # plt.plot(epochs,test_loss_i,label = 'testing loss')
        # plt.ylim(0,1)
        # plt.legend()
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss (Ha)')
        # plt.title('Data size: '+str(training_size[i]))
        # plt.savefig('err_vs_epoch_2layer_2/err_vs_epoch_'+str(training_size[i])+'.pdf')
        # plt.close()

        train_mae_f = trainmae[i][-1]
        train_mse_f = trainmse[i][-1]
        test_loss_f = testloss[i][-1]
        test_mae_f = testmae[i][-1]
        test_mse_f = testmse[i][-1]

        lowest_test_loss.append(min(test_loss_i))
        lowest_test_mae.append(min(test_mae_i))


#print(np.argmin(np.array(final_test_mae)),unit_str[np.argmin(np.array(final_test_mae))])
#print(len(lowest_test_loss))
#print(len(unit_str))
#print(len(N_units_int))
#ind = range(len(training_size))
#unit_str = np.array(unit_str,dtype=str)
#lowest_test_loss = np.array(lowest_test_loss)
#lowest_test_mae = np.array(lowest_test_mae)

#x = 1
#indx_loss = np.argpartition(lowest_test_loss,x)
#indx_mae = np.argpartition(lowest_test_mae,x)

#indx_large = lowest_test_loss > 100
#N_unit_int_arr = np.array(N_units_int)
#print(len(lowest_test_loss))
#print('Configurations with loss > 100:',[list(x) for x in N_unit_int_arr[indx_large]])

# # PLOT WHOLE SWEEP LOSS
# plt.figure()
# plt.bar(ind,lowest_test_loss)
# plt.xticks(ind,unit_str,rotation=90)
# plt.ylim(0,.5)
# plt.title('Whole loss sweep')
# # #plt.show()
#
# # PLOT LOWEST X LOSS
# plt.figure()
# plt.bar(ind[0:x],lowest_test_loss[indx_loss[0:x]])
# plt.xticks(ind[0:x],unit_str[indx_loss[0:x]],rotation=45)
# plt.title('Lowest '+str(x)+' loss')
# #plt.ylim(0,1)
# #plt.show()
#
# # PLOT WHOLE SWEEP MAE
# # plt.figure()
# # plt.bar(ind,lowest_test_mae)
# # plt.xticks(ind,unit_str,rotation=90)
# # plt.ylim(0,1)
# # plt.title('Whole MAE sweep')
# # #plt.show()
#
# #PLOT LOWEST X mae
# plt.figure()
# plt.bar(ind[0:x],lowest_test_mae[indx_mae[0:x]])
# plt.xticks(ind[0:x],unit_str[indx_mae[0:x]],rotation=45)
# plt.title('Lowest '+str(x)+' MAE')
# #plt.ylim(0,1)
# #plt.show()

final_test_loss = np.array(final_test_loss)
final_test_mae = np.array(final_test_mae)

plt.figure()
plt.plot(training_size,final_test_loss,label='test loss')
plt.plot(training_size,final_test_mae,label='test mae')
plt.legend()
plt.ylim(0,1)
plt.show()
#indx_loss = np.argsort(final_test_loss) #np.argpartition(final_test_loss,x)
#indx_mae = np.argsort(final_test_mae) #np.argpartition(final_test_mae,x)

# PLOT WHOLE SWEEP LOSS final
# plt.figure()
# plt.bar(ind,final_test_loss[indx_loss])
# plt.xticks(ind,unit_str[indx_loss],rotation=90)
# #plt.ylim(0,.5)
# plt.title('Whole loss sweep final')
# plt.show()

# PLOT LOWEST X LOSS final
# plt.figure()
# plt.bar(ind[0:x],final_test_loss[indx_loss[0:x]])
# plt.xticks(ind[0:x],unit_str[indx_loss[0:x]],rotation=45)
# plt.title('Lowest '+str(x)+' loss final')
#plt.ylim(0,1)
#plt.show()
#print('lowest loss:',final_test_loss[indx_loss[0:x]])
#print('units:',unit_str[indx_loss[0:x]])

#
# # PLOT WHOLE SWEEP MAE
# # plt.figure()
# # plt.bar(ind,lowest_test_mae)
# # plt.xticks(ind,unit_str,rotation=90)
# # plt.ylim(0,1)
# # plt.title('Whole MAE sweep')
# # #plt.show()
#
#PLOT LOWEST X mae
# plt.figure()
# plt.bar(ind[0:x],final_test_mae[indx_mae[0:x]])
# plt.xticks(ind[0:x],unit_str[indx_mae[0:x]],rotation=45)
# plt.title('Lowest '+str(x)+' MAE final')
# #plt.ylim(0,1)
# plt.show()
#
# # PLOT WHOLE SWEEP LOSS final
# plt.figure()
# plt.bar(ind,(final_test_loss-np.array(final_train_loss))/final_test_loss)
# plt.xticks(ind,unit_str,rotation=90)
# #plt.ylim(0,.5)
# plt.title('Whole loss sweep final')
# plt.show()
