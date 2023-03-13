import tensorflow
import keras
import numpy as np
import sys
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.callbacks import Callback
from matplotlib import pyplot as plt


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.keys = ['loss', 'acc', 'val_loss', 'val_acc']
        self.values = {}
        for k in self.keys:
            self.values['batch_'+k] = []
            self.values['epoch_'+k] = []

    def on_batch_end(self, batch, logs={}):
        for k in self.keys:
            bk = 'batch_'+k
            if k in logs:
                self.values[bk].append(logs[k])

    def on_epoch_end(self, epoch, logs={}):
        for k in self.keys:
            ek = 'epoch_'+k
            if k in logs:
                self.values[ek].append(logs[k])

    def plot(self, keys):
        for key in keys:
            plt.plot(np.arange(len(self.values[key])), np.array(self.values[key]), label=key)
        plt.legend()



# COULOMB MATRICES
inp_data = np.load(sys.argv[1]) # input data should be sys.argv 1, in the form of a .npy file
# ENERGY DATA
U0,U298,H298,G298,Cv298 = np.loadtxt(sys.argv[2],delimiter=',',skiprows=1,usecols=(11,12,13,14,15),unpack=True,comments=None) # prediction data should be in sys.argv 2 in csv file

# SPLIT FOR TESTING VS TRAINING, indicate what E to use
#split = int(U0.shape[0] * 0.2)
energy_type = 'U0'
train_set = U0
test_set = U0
print('Using data for ',energy_type,flush=True)

init_split = .8
i = int(U0.shape[0] * init_split)
all_data = inp_data[:i,:]
all_Es = train_set[:i]
#print(data.shape)

# TRAINING DATA
train_M = inp_data[:i,:]
train_E = train_set[:i]
train_E_arr = np.array([[E] for E in train_E])
print('Training data size: ',train_E.shape[0],flush=True)

# TESTING DATA
test_M = inp_data[i:,:]
test_E = test_set[i:]
test_E_arr = np.array([[E] for E in test_E])
print('Testing data size: ',test_E.shape[0],flush=True)

#N_layers = 3

def train_model_4layer(X_train,y_train,X_test,y_test,N_units,acts,loss_fn,epochs,batch_size):
    model = Sequential()

    N_units1,N_units2,N_units3,N_units4 = N_units
    act1,act2,act3,act4 = acts
    inp_shape = X_train[0].shape[0]

    layer1 = Dense(input_dim = inp_shape ,units=N_units1,activation=act1)
    model.add(layer1)

    layer2 = Dense(units=N_units2,activation=act2)
    model.add(layer2)

    layer3 = Dense(units=N_units3,activation=act3)
    model.add(layer3)

    layer4 = Dense(units=N_units4,activation=act4)
    model.add(layer4)

    #output layer:
    out_layer = Dense(units=1,activation='linear')
    model.add(out_layer)

    model.compile(loss=loss_fn, optimizer=Adam(), metrics=["mean_squared_error",'mae'])

    # TRAIN THE MODEL
    #history = LossHistory()
    fit = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test,y_test), verbose=False,batch_size=batch_size),#callbacks=[history], verbose=True)
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

    return train_loss,train_mse,train_mae,test_loss,test_mse,test_mae

#units = [29,25,20,15,10,5,1]
units = [35,25,15,5]
for N_unit1 in units:
    for N_unit2 in units:
        for N_unit3 in units:
            for N_unit4 in units:

                N_units = [N_unit1,N_unit2,N_unit3,N_unit4]
                acts = ['relu','relu','relu','relu']
                loss_fn = 'mean_squared_error'
                epochs = 20
                batch_size = 32
                print('Number of layers: ', 4,flush=True)
                print('Number of units in layer [1,2,3,4]: ',N_units,flush=True)
                print('Activation function in layer [1,2,3,4]: ',acts,flush=True)
                print('Loss function: ',loss_fn,flush=True)
                print('Number of epochs: ',epochs,flush=True)
                print('Batch size: ',batch_size,flush=True)
                print('',flush=True)

                train_model_4layer(train_M,train_E_arr,test_M,test_E_arr,N_units,acts,loss_fn,epochs,batch_size)
                print('',flush=True)
