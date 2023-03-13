import tensorflow
import keras
import numpy as np
import sys
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.callbacks import Callback
import matplotlib.pyplot as plt


def train_model_5layer(X_train,y_train,X_test,y_test,N_units,acts,loss_fn,epochs,batch_size):
    model = Sequential()

    N_units1,N_units2,N_units3,N_units4,N_units5 = N_units
    act1,act2,act3,act4,act5 = acts
    inp_shape = X_train[0].shape[0]

    layer1 = Dense(input_dim = inp_shape ,units=N_units1,activation=act1)
    model.add(layer1)

    layer2 = Dense(units=N_units2,activation=act2)
    model.add(layer2)

    layer3 = Dense(units=N_units3,activation=act3)
    model.add(layer3)

    layer4 = Dense(units=N_units4,activation=act4)
    model.add(layer4)

    layer5 = Dense(units=N_units5,activation=act5)
    model.add(layer5)

    #output layer:
    out_layer = Dense(units=1,activation='linear')
    model.add(out_layer)

    model.compile(loss=loss_fn, optimizer=Adam(), metrics=["mean_squared_error",'mae'])

    # TRAIN THE MODEL
    #history = LossHistory()
    fit = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test,y_test), verbose=False,batch_size=batch_size),#callbacks=[history], verbose=True)
    #print(fit[0].history)
    val_loss_arr = fit[0].history['val_loss']
    val_mse_arr = fit[0].history['val_mean_squared_error']
    val_mae_arr = fit[0].history['val_mean_absolute_error']
    train_loss_arr = fit[0].history['loss']
    train_mse_arr = fit[0].history['mean_squared_error']
    train_mae_arr = fit[0].history['mean_absolute_error']
    for i in range(epochs):
        print('Epoch ',i)
        print('Training loss: ',train_loss_arr[i],flush=True)
        print('Training MSE:  ',train_mse_arr[i],flush=True)
        print('Training MAE:  ',train_mae_arr[i],flush=True)
        print('Testing loss:  ',val_loss_arr[i],flush=True)
        print('Testing MSE:   ',val_mse_arr[i],flush=True)
        print('Testing MAE:   ',val_mae_arr[i],flush=True)
        print('',flush=True)

    train_eval=model.evaluate(X_train,y_train,verbose=0)
    train_loss = train_eval[0]
    train_mse = train_eval[1]
    train_mae = train_eval[2]
    print('Training loss: ',train_loss,flush=True)
    print('Training MSE:  ',train_mse,flush=True)
    print('Training MAE:  ',train_mae,flush=True)
    print('',flush=True)

    test_eval = model.evaluate(X_test,y_test,verbose=0)
    test_loss = test_eval[0]
    test_mse = test_eval[1]
    test_mae = test_eval[2]
    print('Testing loss: ',test_loss,flush=True)
    print('Testing MSE:  ',test_mse,flush=True)
    print('Testing MAE:  ',test_mae,flush=True)
    print("",flush=True)

    plt.figure()
    plt.plot(range(epochs),train_loss_arr)
    plt.plot(range(epochs),val_loss_arr)
    plt.ylim(0,1)
    plt.savefig('err_vs_epoch_3.pdf')

    #plt.figure()


    return model


#COULOMB MATRICES
inp_data = np.load(sys.argv[1]) # input data should be sys.argv 1, in the form of a .npy file
# ENERGY DATA
U0,U298,H298,G298,Cv298 = np.loadtxt(sys.argv[2],delimiter=',',skiprows=1,usecols=(11,12,13,14,15),unpack=True,comments=None) # prediction data should be in sys.argv 2 in csv file
smiles = np.loadtxt(sys.argv[2],delimiter=',',skiprows=1,usecols=(0),unpack=True,comments=None,dtype=str)

# SPLIT FOR TESTING VS TRAINING, indicate what E to use
#split = int(U0.shape[0] * 0.2)
energy_type = 'H298'
train_set = H298
test_set = H298
print('Using data for ',energy_type,flush=True)

N_units = [25,15,5,5,15]
# 35 5 35 5 25, 0.01680047
# '25 15 5 5 15', 0.01651607 ***
# '15 5 5 25 25', 0.01863627
# '5 5 5 25 25', 0.02059377
#splits = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#for init_split in splits:
    #init_split = .8 # how much of dataset to use
#j = int(U0.shape[0] * init_split)
print('Total dataset size: ', len(U0))
all_data = inp_data[:,:]
all_Es = train_set[:]
#print(data.shape)

train_split = .8
i = int(all_Es.shape[0] * train_split)
# TRAINING DATA
train_M = all_data[:i,:]
train_E = all_Es[:i]
train_E_arr = np.array([[E] for E in train_E])
train_smiles = smiles[:i]
print('Training data size: ',train_E.shape[0],flush=True)

# TESTING DATA
test_M = all_data[i:,:]
test_E = all_Es[i:]
test_E_arr = np.array([[E] for E in test_E])
test_smiles = smiles[i:]
print('Testing data size: ',test_E.shape[0],flush=True)

#N_units = [N_unit1,N_unit2,N_unit3]
acts = ['relu','relu','relu','relu','relu']
loss_fn = 'mean_squared_error'
epochs = 20
batch_size = 32
print('Number of layers: ', 5,flush=True)
print('Number of units in layer [1,2,3,4,5]: ',N_units,flush=True)
print('Activation function in layer [1,2,3,4,5]: ',acts,flush=True)
print('Loss function: ',loss_fn,flush=True)
print('Number of epochs: ',epochs,flush=True)
print('Batch size: ',batch_size,flush=True)
print('',flush=True)

model = train_model_5layer(train_M,train_E_arr,test_M,test_E_arr,N_units,acts,loss_fn,epochs,batch_size)
pred_train = model.predict(train_M)
pred_test = model.predict(test_M)

print('Test E shape',test_E_arr.shape)
print('Pred test E shape',pred_test.shape)
print('smiles test',test_smiles.shape)
print("train E shape",train_E_arr.shape)
print("pred train E shape",pred_train.shape)
print('smiles train',train_smiles.shape)

output_file_train = sys.argv[3]
output_file_test = sys.argv[4]
f1 = open(output_file_train,'w')
f1.write('SMILES,U0 (DFT),U0 (predicted),Signed error,Absolute Error\n')
for i in range(0,train_E_arr.shape[0]-1):
    f1.write(train_smiles[i]+','+str(train_E[i])+','+str(pred_train[i][0])+','+str(pred_train[i][0]-train_E[i])+','+str(np.abs(pred_train[i][0]-train_E[i]))+'\n')
f1.close()

f2 = open(output_file_test,'w')
f2.write('SMILES,U0 (DFT),U0 (predicted),Signed error,Absolute Error\n')
for i in range(0,test_E_arr.shape[0]-1):
    f2.write(test_smiles[i]+','+str(test_E[i])+','+str(pred_test[i][0])+','+str(pred_test[i][0]-test_E[i])+','+str(np.abs(pred_test[i][0]-test_E[i]))+'\n')
f2.close()
