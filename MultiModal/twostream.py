import scipy.io as sio
import numpy as np

import tensorflow as tf
import os
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"]="2"
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
np.random.seed(666)

def loadData():
   
    msdata = np.load('/home/SharedData/saurabh/HS/multispectralData.npy')
    pcdata = np.load('/home/SharedData/saurabh/HS/panchromaticData.npy')
    labels = np.load('/home/SharedData/saurabh/HS/labelsData.npy')
    
    return msdata, pcdata, labels


msdata, pcdata, gt = loadData()

print("ORIGINAL SHAPES - ")
print(msdata.shape)
print(pcdata.shape)
print(gt.shape)

msdata=np.transpose(msdata,(0,2,3,1))
pcdata=np.transpose(pcdata,(0,2,3,1))
gt=np.transpose(gt,(1,0))
gt=np.ravel(gt)

print("TRANSPOSED SHAPES - ")
print(msdata.shape)
print(pcdata.shape)
print(gt.shape)


msdata = msdata.astype(np.float32)
pcdata = pcdata.astype(np.float32)

it2 = 5000
for it1 in range(0,80000,5000):

	print(it1,it2)
	msdata[it1:it2,:,:,:] = (msdata[it1:it2,:,:,:] - msdata[it1:it2,:,:,:].mean(axis=(0, 1, 2), keepdims=True))/(msdata[it1:it2,:,:,:].std(axis=(0, 1,2), keepdims=True))
	pcdata[it1:it2,:,:,:] = (pcdata[it1:it2,:,:,:] - pcdata[it1:it2,:,:,:].mean(axis=(0,1, 2), keepdims=True))/(pcdata[it1:it2,:,:,:].std(axis=(0,1,2), keepdims=True))
	it2 += 5000

print(msdata.shape, pcdata.shape)



from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(gt)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
gt = onehot_encoder.fit_transform(integer_encoded)

print(msdata.shape,pcdata.shape,gt.shape)

from sklearn.model_selection import train_test_split

(Xtrain_one, Xtest_one, Ytrain, Ytest) = train_test_split(msdata, gt, random_state = 666, test_size= 0.75)
(Xtrain_two, Xtest_two, Ytrain, Ytest) = train_test_split(pcdata, gt, random_state = 666, test_size = 0.75 )


from keras.models import Model
from keras import optimizers
from keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten, Input, 
                          concatenate, Add, BatchNormalization, Dropout, Lambda)


#modelinput1 = Input(shape=(64,64,4), name='MultiModal_Input_1')
#modelinput2 = Input(shape=(256,256,1), name='Multimodal_Input_2')


#MS Network

modelinput1 = Input(shape=(64,64,4), name='MultiModal_Input_1')

conv11 = Conv2D(128, (5, 5), activation='relu')(modelinput1)
conv11 = MaxPooling2D((2,2))(conv11)

conv12 = Conv2D(128, (5, 5), activation='relu' , padding = 'same')(conv11)
conv12 = MaxPooling2D((2,2))(conv12)
conv12 = BatchNormalization()(conv12)

conv13 = Conv2D(64, (5, 5), activation='relu' , padding = 'same')(conv12)
conv13 = MaxPooling2D((3,3))(conv13)
conv13 = BatchNormalization()(conv13)
conv13 = Dropout(0.1)(conv13)

model1 = Flatten()(conv13)
model1 = Dense(448, activation='relu')(model1)
model1 = Dropout(0.1)(model1)
model1 = Dense(200, activation='relu', name='MultiModal_Output_1')(model1)


#PC Network

modelinput2 = Input(shape=(256,256,1), name='Multimodal_Input_2')

conv21 = Conv2D(128, (5, 5), activation='relu')(modelinput2)
conv21 = MaxPooling2D((3,3))(conv21)

conv22 = Conv2D(128, (5, 5), activation='relu' , padding = 'same')(conv21)
conv22 = MaxPooling2D((2,2))(conv22)
conv22 = BatchNormalization()(conv22)

conv23 = Conv2D(64, (5, 5), activation='relu' , padding = 'same')(conv22)
conv23 = MaxPooling2D((2,2))(conv23)
conv23 = BatchNormalization()(conv23)
conv23 = Dropout(0.1)(conv23)

conv24 = Conv2D(64, (5, 5), activation='relu' , padding = 'same')(conv23)
conv24 = MaxPooling2D((2,2))(conv24)
conv24 = BatchNormalization()(conv24)
conv24 = Dropout(0.1)(conv24)

model2 = Flatten()(conv24)
model2 = Dense(512, activation='relu')(model2)
model2 = Dropout(0.1)(model2)
model2 = Dense(256, activation='relu')(model2)
model2 = Dropout(0.1)(model2)
model2 = Dense(200, activation='relu', name='MultiModal_Output_2')(model2)



modelfirst = Model(inputs=modelinput1, outputs=model1)
modelsecond = Model(inputs=modelinput2, outputs=model2)

combined = concatenate([modelfirst.output, modelsecond.output] , name = 'Classifier_Input')

combinedoutput= Dense(1024,activation='relu')(combined)
combinedoutput = BatchNormalization()(combinedoutput)
combinedoutput = Dropout(0.5)(combinedoutput)

combinedoutput= Dense(1024,activation='relu')(combinedoutput)
combinedoutput = BatchNormalization()(combinedoutput)
combinedoutput = Dropout(0.5)(combinedoutput)

#Set temperature parameter for softmax
combinedoutput = Lambda(lambda x: x / 2)(combinedoutput)
combinedoutput= Dense(8,activation='softmax', name='Classifier_Output')(combinedoutput)

#model = Model(inputs=[modelinput1,modelinput2], outputs=combinedoutput)
model=load_model('/home/SharedData/Avinandan/TeacherStudentExperiments/MultiModal/twostream.h5')
print(model.summary())
sgd = optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-6)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('/home/SharedData/Avinandan/TeacherStudentExperiments/MultiModal/twostream.h5', monitor='val_acc', verbose=1
                            ,save_best_only=True, mode='max')
callback_list = [checkpoint]
#model.fit(
#    [Xtrain_one, Xtrain_two], Ytrain, callbacks=callback_list,
#    validation_data=([Xtest_one, Xtest_two],Ytest),
#    epochs=300, verbose = 2)

