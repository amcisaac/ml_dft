import tensorflow
import keras
import numpy as np
import sys
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.callbacks import Callback

def train_model_3layer(X_train,y_train,X_test,y_test,N_units,acts,loss_fn,epochs,batch_size):
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
    val_loss_arr = fit[0].history['val_loss']
    val_mse_arr = fit[0].history['val_mean_squared_error']
    val_mae_arr = fit[0].history['val_mean_absolute_error']
    train_loss_arr = fit[0].history['loss']
    train_mse_arr = fit[0].history['mean_squared_error']
    train_mae_arr = fit[0].history['mean_absolute_error']
    # for i in range(epochs):
    #     print('Epoch ',i)
    #     print('Training loss: ',train_loss_arr[i],flush=True)
    #     print('Training MSE:  ',train_mse_arr[i],flush=True)
    #     print('Training MAE:  ',train_mae_arr[i],flush=True)
    #     print('Testing loss:  ',val_loss_arr[i],flush=True)
    #     print('Testing MSE:   ',val_mse_arr[i],flush=True)
    #     print('Testing MAE:   ',val_mae_arr[i],flush=True)
    #     print('',flush=True)

    train_eval=model.evaluate(X_train,y_train,verbose=0)
    train_loss = train_eval[0]
    train_mse = train_eval[1]
    train_mae = train_eval[2]
    # print('Training loss: ',train_loss,flush=True)
    # print('Training MSE:  ',train_mse,flush=True)
    # print('Training MAE:  ',train_mae,flush=True)
    # print('',flush=True)

    test_eval = model.evaluate(X_test,y_test,verbose=0)
    test_loss = test_eval[0]
    test_mse = test_eval[1]
    test_mae = test_eval[2]
    # print('Testing loss: ',test_loss,flush=True)
    # print('Testing MSE:  ',test_mse,flush=True)
    # print('Testing MAE:  ',test_mae,flush=True)
    # print("",flush=True)

    return train_loss,train_mse,train_mae,test_loss,test_mse,test_mae

def xval_learning_alg(data, Es, k,N_units,acts,loss_fn,epochs,batch_size):
    d,n=data.shape
    print('Input data shape:',data.shape,flush=True)
    print('Energy data shape:',Es.shape,flush=True)
    #print(d,n)
    #print(k)
    #print(Es[0:5])
    split_data=np.array_split(data,k,axis=0) # transposed from what we had in class
    split_labels = np.array_split(Es,k,axis=0)
    # print(len(split_data))
    # print(len(split_labels))
    #print(split_data[0].shape)
    #print(split_labels[0].shape)

    print('Training data size:',data.shape[0] - split_data[0].shape[0],flush=True)
    print('Testing data size:',split_data[0].shape[0],flush=True)
    # print(Es)
    # print(split_labels)
    avg_test_err=0
    avg_train_err = 0
    avg_test_mae = 0
    avg_train_mae = 0
    for j in range(0,k):
        print('Cross-validation round ', j,flush=True)
        # process data
        if j != 0 and j != k-1:
            test1 = np.concatenate(split_data[0:j],axis=0)
            test2 = np.concatenate(split_data[j+1:],axis=0)
            train_data = np.concatenate((test1,test2),axis=0)

            test1l = np.concatenate(split_labels[0:j],axis=0)
            test2l = np.concatenate(split_labels[j+1:],axis=0)
            train_Es = np.concatenate((test1l,test2l),axis=0)

        elif j == 0:
            train_data = np.concatenate(split_data[j+1:],axis=0)
            train_Es = np.concatenate(split_labels[j+1:],axis=0)


        elif j == k-1:
            #print('end')
            train_data = np.concatenate(split_data[0:j],axis=0)
            train_Es = np.concatenate(split_labels[0:j],axis=0)
        test_data = split_data[j]
        test_Es = split_labels[j]
        test_E_arr = np.array([[E] for E in test_Es])
        train_E_arr =np.array([[E] for E in train_Es])

        # print(test_data.shape)
        # print(train_data.shape)
        # print(test_E_arr.shape)
        # print(train_E_arr.shape)

        train_loss,train_mse,train_mae,test_loss,test_mse,test_mae=train_model_3layer(train_data,train_E_arr,test_data,test_E_arr,N_units,acts,loss_fn,epochs,batch_size)


        print('Training loss: ',train_loss,flush=True)
        print('Training MSE:  ',train_mse,flush=True)
        print('Training MAE:  ',train_mae,flush=True)
        print('',flush=True)
        print('Testing loss: ',test_loss,flush=True)
        print('Testing MSE:  ',test_mse,flush=True)
        print('Testing MAE:  ',test_mae,flush=True)
        print('',flush=True)

        avg_test_err += test_loss
        avg_train_err += train_loss
        avg_test_mae += test_mae
        avg_train_mae += train_mae
        #score_j=eval_classifier(learner,D_minus_j,lab_min_j,split_data[j],split_labels[j])
        #avg_score = avg_score + score_j
    return avg_test_err/k, avg_train_err/k,avg_test_mae/k,avg_train_mae/k


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
N_units = [25, 15, 5, 5, 15]
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

avg_test_err,avg_train_err,avg_test_mae,avg_train_mae = xval_learning_alg(inp_data, U0,5,N_units,acts,loss_fn,epochs,batch_size)
print("Cross-validation training loss: ",avg_train_err,flush=True)
print("Cross-validation training MAE: ",avg_train_mae,flush=True)
print("Cross-validation testing loss: ",avg_test_err,flush=True)
print("Cross-validation testing MAE: ",avg_test_mae,flush=True)
print('',flush=True)
