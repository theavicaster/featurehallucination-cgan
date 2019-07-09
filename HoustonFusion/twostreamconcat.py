import scipy.io as sio
import numpy as np

import tensorflow as tf
import os
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"]="2"
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
np.random.seed(666)

def loadData():
   
    data = sio.loadmat('/home/SharedData/Avinandan/HoustonFusion/dfc2013.mat')['data_hs']
    labels = sio.loadmat('/home/SharedData/Avinandan/HoustonFusion/dfc2013.mat')['test']
    
    return data, labels


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createPatches(X, y, windowSize=17, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

data, labels = loadData()

print(data.shape, labels.shape)
#print(gt.shape)




data_mean = data.mean(axis=(0, 1), keepdims=True)
data_std = data.std(axis=(0, 1), keepdims=True)
data = (data - data_mean)/(data_std)
print(data.dtype)
data = (data - data.mean(axis=(0, 1), keepdims=True))/(data.std(axis=(0, 1), keepdims=True))
data = data.astype(np.float32)
print(data.dtype)



X, Y = createPatches(data, labels, windowSize=3, removeZeroLabels = False)

data = None
labels = None

print(X.shape,Y.shape)




from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y = onehot_encoder.fit_transform(integer_encoded)

print(X.shape,Y.shape)

X_one = (X[:,:,:,0:100])
X_two = (X[:,:,:,100:144])

X = None

print(X_one.shape, X_two.shape , "100 channels and 44 channels split")







from sklearn.model_selection import train_test_split

(Xtrain_one, Xtest_one, Ytrain, Ytest) = train_test_split(X_one, Y, random_state = 666,test_size = 0.5)
X_one = None

(Xtrain_two, Xtest_two, Ytrain, Ytest) = train_test_split(X_two, Y, random_state = 666, test_size = 0.5)
X_two=None

print(Xtrain_one.shape,Ytrain.shape)






from keras.models import Model
from keras import optimizers
from keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten, Input, 
                          concatenate, Add, BatchNormalization, Dropout, Lambda)


modelinput1 = Input(shape=(3, 3, 100), name='Houston_Fusion_Input_1')
modelinput2 = Input(shape=(3, 3, 44), name='Houston_Fusion_Input_2')

conv11 = Conv2D(128, (3, 3), activation='relu')(modelinput1)
conv11 = MaxPooling2D((1, 1))(conv11)

conv12 = Conv2D(128, (3, 3), activation='relu')(modelinput2)
conv12 = MaxPooling2D((1, 1))(conv12)

#model1 = conv1(modelinput1)
#model2 = conv1(modelinput2)


conv2 = Conv2D(128, (1, 1), activation='relu', padding='same')

model1 = conv2(conv11)
model2 = conv2(conv12)

bn2 = BatchNormalization()
model1 = bn2(model1)
model2 = bn2(model2)

conv3 = Conv2D(128, (1, 1), activation='relu', padding='same')
model1 = conv3(model1)
model2 = conv3(model2)

bn3 = BatchNormalization()
model1 = bn3(model1)
model2 = bn3(model2)


conv4 = Conv2D(128, (1, 1), activation='relu', padding='same')
model1 = conv4(model1)
model2 = conv4(model2)


bn4 = BatchNormalization()
model1 = bn4(model1)
model2 = bn4(model2)

conv5 = Conv2D(128, (1, 1), activation='relu', padding='same')
model1 = conv5(model1)
model2 = conv5(model2)

bn5 = BatchNormalization()
model1 = bn5(model1)
model2 = bn5(model2)


model1 = Conv2D(64, (1, 1), activation='relu')(model1)
model1 = BatchNormalization()(model1)

model2 = Conv2D(64, (1, 1), activation='relu')(model2)
model2 = BatchNormalization()(model2)

model1 = Conv2D(64, (1, 1), activation='relu')(model1)
model1 = BatchNormalization()(model1)
model1 = Dropout(0.1)(model1)

model2 = Conv2D(64, (1, 1), activation='relu')(model2)
model2 = BatchNormalization()(model2)
model2 = Dropout(0.1)(model2)


model1 = Flatten()(model1)
model1 = Dense(512, activation='relu')(model1)
model1 = Dense(200, activation='relu', name='Houston_Fusion_Output_1')(model1)

model2 = Flatten()(model2)
model2 = Dense(512, activation='relu')(model2)
model2 = Dense(200, activation='relu', name='Houston_Fusion_Output_2')(model2)

modelfirst = Model(inputs=modelinput1, outputs=model1)
modelsecond = Model(inputs=modelinput2, outputs=model2)

combined = concatenate([modelfirst.output, modelsecond.output] , name = 'Classifier_Input')

combinedoutput= Dense(1024,activation='relu')(combined)
combinedoutput = BatchNormalization()(combinedoutput)
combinedoutput = Dropout(0.1)(combinedoutput)

combinedoutput= Dense(1024,activation='relu')(combinedoutput)
combinedoutput = BatchNormalization()(combinedoutput)
combinedoutput = Dropout(0.1)(combinedoutput)

#Set temperature parameter for softmax
combinedoutput = Lambda(lambda x: x / 2)(combinedoutput)
combinedoutput= Dense(16,activation='softmax', name='Classifier_Output')(combinedoutput)

model = Model(inputs=[modelinput1,modelinput2], outputs=combinedoutput)
print(model.summary())
sgd = optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-6)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('/home/SharedData/Avinandan/TeacherStudentExperiments/HoustonFusion/twostreamconcat.h5', monitor='val_acc', verbose=1
                            ,save_best_only=True, mode='max')
callback_list = [checkpoint]
model.fit(
    [Xtrain_one, Xtrain_two], Ytrain, validation_split = 0.25, callbacks=callback_list,
    validation_data=([Xtest_one, Xtest_two],Ytest),
    epochs=300, verbose = 2)


