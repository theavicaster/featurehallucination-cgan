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
   
    data = sio.loadmat('/home/SharedData/Avinandan/PaviaU/PaviaU.mat')['paviaU']
    labels = sio.loadmat('/home/SharedData/Avinandan/PaviaU/PaviaU_gt.mat')['paviaU_gt']
    
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

data, gt = loadData()

print(data.shape)
print(gt.shape)

data_mean = data.mean(axis=(0, 1), keepdims=True)
data_std = data.std(axis=(0, 1), keepdims=True)
data = (data - data_mean)/(data_std)
print(data.dtype)
data = (data - data.mean(axis=(0, 1), keepdims=True))/(data.std(axis=(0, 1), keepdims=True))
data = data.astype(np.float32)
print(data.dtype)

X, Y = createPatches(data, gt, windowSize=15)

print(X.shape,Y.shape)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y = onehot_encoder.fit_transform(integer_encoded)

print(X.shape,Y.shape)

X_one = (X[:,:,:,0:50])
X_two = (X[:,:,:,50:103])

print(X_one.shape, X_two.shape , "50 channels and 53 channels split")

from sklearn.model_selection import train_test_split

(Xtrain_one, Xtest_one, Ytrain, Ytest) = train_test_split(X_one, Y, random_state = 666)
(Xtrain_two, Xtest_two, Ytrain, Ytest) = train_test_split(X_two, Y, random_state = 666)


from keras.models import Model
from keras import optimizers
from keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten, Input, 
                          concatenate, Add, BatchNormalization, Dropout, Lambda, Average)


modelinput1 = Input(shape=(15, 15, 50), name='Pavia_Univ_Input_1')
modelinput2 = Input(shape=(15, 15, 53), name='Pavia_Univ_Input_2')

conv11 = Conv2D(128, (5, 5), activation='relu')(modelinput1)
conv11 = MaxPooling2D((5, 5))(conv11)

conv12 = Conv2D(128, (5, 5), activation='relu')(modelinput2)
conv12 = MaxPooling2D((5, 5))(conv12)

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
model1 = Dense(200, activation='relu', name='Pavia_Univ_Output_1')(model1)

model2 = Flatten()(model2)
model2 = Dense(512, activation='relu')(model2)
model2 = Dense(200, activation='relu', name='Pavia_Univ_Output_2')(model2)



classifier1= Dense(1024,activation='relu')(model1)
classifier1 = BatchNormalization()(classifier1)
classifier1 = Dropout(0.1)(classifier1)

classifier2= Dense(1024,activation='relu')(model2)
classifier2 = BatchNormalization()(classifier2)
classifier2 = Dropout(0.1)(classifier2)

classifier1= Dense(1024,activation='relu')(classifier1)
classifier1 = BatchNormalization()(classifier1)
classifier1 = Dropout(0.1)(classifier1)

classifier2= Dense(1024,activation='relu')(classifier2)
classifier2 = BatchNormalization()(classifier2)
classifier2 = Dropout(0.1)(classifier2)


#Set temperature parameter for softmax
classifier1 = Lambda(lambda x: x / 2)(classifier1)
classifier1= Dense(9,activation='softmax', name='Classifier_Output_1')(classifier1)

classifier2 = Lambda(lambda x: x / 2)(classifier2)
classifier2= Dense(9,activation='softmax', name='Classifier_Output_2')(classifier2)

classifierfinal = Average()([classifier1,classifier2])

model = Model(inputs=[modelinput1,modelinput2],outputs=classifierfinal)
print(model.summary())
sgd = optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-6)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('/home/SharedData/Avinandan/TeacherStudentExperiments/PaviaUniv/2N/Average/twostreamaverage.h5', monitor='val_acc', verbose=1
                            ,save_best_only=True, mode='max')
callback_list = [checkpoint]
model.fit(
    [Xtrain_one, Xtrain_two], Ytrain, validation_split = 0.25, callbacks=callback_list,
    validation_data=([Xtest_one, Xtest_two],Ytest),
    epochs=300, verbose = 2)