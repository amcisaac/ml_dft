import numpy as np
import matplotlib.pyplot as plt
import sys

y = int(sys.argv[-1])

N_units = []
epochs = range(0,20)
trainloss = []
trainmae = []
trainmse = []
testloss = []
testmae = []
testmse = []

N_units_int = []

lowest_test_loss = []
lowest_test_mae = []
unit_str = []

final_test_loss = []
final_test_mae = []
final_train_loss = []
final_train_mae = []

test = 0
for inpfile in sys.argv[1:-1]:
    #inpfile = sys.argv[1]
    f = open(inpfile,'r')
    filelines = f.readlines()
    f.close()
    print(inpfile)

    N = 0
    print('here')
    for line in filelines:
        if len(line.split()) > 2 and line.split()[2] == 'units':
            #print(line)
            unit_list = [x.strip('[],') for x in line.split()[-y:]]
            #print(unit_list)
            print(len(unit_list))
            unit_list_int = [int(x.strip('[],')) for x in line.split()[-y:]]
            N_units.append(unit_list)
            N_units_int.append(unit_list_int)
            test += 1
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
    print(len(N_units))
    print(len(trainloss))

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


        # plt.figure()
        # plt.plot(epochs,train_loss_i,label = 'training loss')
        # plt.plot(epochs,test_loss_i,label = 'testing loss')
        # plt.ylim(0,1)
        # plt.legend()
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss (Ha)')
        # plt.title('Number of units: '+' '.join(N_units[i]))
        # plt.savefig('err_vs_epoch_5layer_35/err_vs_epoch_'+'_'.join(N_units[i])+'.pdf')
        # plt.close()

        train_mae_f = trainmae[i][-1]
        train_mse_f = trainmse[i][-1]
        test_loss_f = testloss[i][-1]
        test_mae_f = testmae[i][-1]
        test_mse_f = testmse[i][-1]

        lowest_test_loss.append(min(test_loss_i))
        lowest_test_mae.append(min(test_mae_i))

ind = np.array(range(len(unit_str)))
print(len(final_test_loss))
#print(test)
#print(ind)
x = 6
final_test_loss = np.array(final_test_loss)
final_test_mae = np.array(final_test_mae)
final_train_loss = np.array(final_train_loss)
final_train_mae = np.array(final_train_mae)
indx_loss = np.argsort(final_test_loss) #np.argpartition(final_test_loss,x)
indx_mae = np.argsort(final_test_mae) #np.argpartition(final_test_mae,x)
#print(indx_loss)
unit_str = np.array(unit_str,dtype=str)
#print(unit_str)
print(len(N_units))
print(len(unit_str))
print(final_test_loss.shape)
#
# # PLOT WHOLE SWEEP LOSS final
# plt.figure()
# plt.bar(ind,final_test_loss[indx_loss])
# plt.xticks(ind,unit_str,rotation=90)
# #plt.ylim(0,.5)
# plt.title('Whole loss sweep final')
# plt.show()

#print(x)
#print(ind[0:x])
print(final_test_loss[indx_loss[0:x]])
#print(unit_str)
# PLOT LOWEST X LOSS final
w = 0.2
m = 0.1
plt.figure()
plt.bar(ind[0:x]-m,final_test_loss[indx_loss[0:x]],width=w,label='Testing')
plt.bar(ind[0:x]+m,final_train_loss[indx_loss[0:x]],width=w,label = 'Training')
plt.xticks(ind[0:x],unit_str[indx_loss[0:x]],rotation=45)
plt.legend()
plt.title('Lowest '+str(x)+' loss final')
#plt.ylim(0,1)
#plt.show()
#

#PLOT LOWEST X mae
plt.figure()
plt.bar(ind[0:x]-m,final_test_mae[indx_mae[0:x]],width=w,label='Testing')
plt.bar(ind[0:x]+m,final_train_mae[indx_mae[0:x]],width=w,label='Training')
plt.xticks(ind[0:x],unit_str[indx_mae[0:x]],rotation=45)
plt.title('Lowest '+str(x)+' MAE final')
plt.legend()
#plt.ylim(0,1)
plt.show()

# # PLOT WHOLE SWEEP LOSS final
# plt.figure()
# plt.bar(ind,(final_test_loss-np.array(final_train_loss))/final_test_loss)
# plt.xticks(ind,unit_str,rotation=90)
# #plt.ylim(0,.5)
# plt.title('Whole loss sweep final')
# plt.show()
