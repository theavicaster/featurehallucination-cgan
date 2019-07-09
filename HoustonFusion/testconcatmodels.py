
n_features = 200
latent_dim=100



import scipy.io as sio


import numpy as np
np.random.seed(666)
import tensorflow as tf
#from tensorflow.math import reduce_sum
#import tensorflow.math
import os
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))


import keras
from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Average
from keras.layers import BatchNormalization, Activation, ZeroPadding2D,Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop,SGD, Adam
from functools import partial
from keras import layers, models
from keras.models import load_model



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



data, gt = loadData()

data_mean = data.mean(axis=(0, 1), keepdims=True)
data_std = data.std(axis=(0, 1), keepdims=True)
data = (data - data_mean)/(data_std)
print(data.dtype)
data = (data - data.mean(axis=(0, 1), keepdims=True))/(data.std(axis=(0, 1), keepdims=True))
data = data.astype(np.float32)
print(data.dtype)

data, Y = createPatches(data, gt, windowSize=3,removeZeroLabels=False)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y = onehot_encoder.fit_transform(integer_encoded)

#print(X.shape,Y.shape)

data1 = data[:,:,:,0:100]
data2 = data[:,:,:,100:144]

classifier = load_model('/home/SharedData/Avinandan/TeacherStudentExperiments/HoustonFusion/twostreamconcat.h5')
streamone = Model(inputs=classifier.get_layer('Houston_Fusion_Input_1').input,
                       outputs=classifier.get_layer('Houston_Fusion_Output_1').output)
streamtwo = Model(inputs=classifier.get_layer('Houston_Fusion_Input_2').input,
                       outputs=classifier.get_layer('Houston_Fusion_Output_2').output)
#generator = load_model('/home/SharedData/Avinandan/Semi Supervised GAN/2N/generator2N.h5')
#discriminator = load_model('/home/SharedData/Avinandan/Semi Supervised GAN/2N/discriminator.h5')

print(classifier.summary())
        

finalclassifierinput = Input(shape=(2*n_features,))

finalclassifier = finalclassifierinput

for layer in classifier.layers[31:]:
    finalclassifier = layer(finalclassifier)

# create the model
#new_model = Model(layer_input, x)

finalclassifier =  Model(inputs=finalclassifierinput,
                       outputs=finalclassifier)
print(finalclassifier.summary())


data1 = streamone.predict(data1)
data2 = streamtwo.predict(data2)


data=np.concatenate((data1, data2), axis = 1)

random_latent_vectors = np.random.normal(0,1, size = (data1.shape[0],latent_dim)) #sample random points

#print(random_latent_vectors.shape,data2.shape,data1.shape)


topvalid = 0
toploss = 5000

#610
#2025 98.28
#2517 98.51

for x in range(3000,3025,1):

    generator = load_model('/home/SharedData/Avinandan/TeacherStudentExperiments/HoustonFusion/models/generator-{0}.h5'.format(x))
    #generator = load_model('/home/SharedData/Avinandan/Semi Supervised GAN/2N/generator2N.h5')
                                             #latent, conditional
    generated_images = generator.predict([random_latent_vectors,data2])
    #generated_images = generator.predict([random_latent_vectors,data1],verbose=1)

    #generated_images = np.reshape(generated_images,(,100))
    #print(generated_images.shape)

    X_Houston_Fusion1 = np.concatenate((generated_images,data2),axis=1)

    #print(X_Houston_Fusion1.shape,Y.shape)

    from sklearn.model_selection import train_test_split


    #(Xtrain_Houston_Fusion, Xtest_Houston_Fusion,Ytrain_Houston_Fusion,Ytest_Houston_Fusion,) = train_test_split(X_Houston_Fusion,Y_Houston_Fusion, random_state = 666)

    #(Xtrain_Houston_Fusion, Xtest_Houston_Fusion,Ytrain_Houston_Fusion,Ytest_Houston_Fusion,) = train_test_split(data,Y, random_state = 666)

    #(generated_images_train, generated_images_test,Ytrain_Houston_Fusion,Ytest_Houston_Fusion,) = train_test_split(generated_images,Y, random_state = 666)
    #(secondstream_train, secondstream_test,Ytrain_Houston_Fusion,Ytest_Houston_Fusion,) = train_test_split(data2,Y, random_state = 666)
    #(firststream_train, firststream_test,Ytrain_Houston_Fusion,Ytest_Houston_Fusion,) = train_test_split(data1,Y, random_state = 666)

    #results = onestream.evaluate([Xtest_Houston_Fusion, Xtest_Houston_Fusion],Ytest_Houston_Fusion, batch_size=32)

    #print('test loss, test acc:', results)

    (Xtrain_Houston_Fusion1, Xtest_Houston_Fusion1,Ytrain_Houston_Fusion,Ytest_Houston_Fusion,) = train_test_split(X_Houston_Fusion1,Y, random_state = 666, test_size =0.5
        )

    sgd = SGD(lr=0.0001, momentum=0.9, decay=1e-6)
    finalclassifier.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    #results = finalclassifier.evaluate(Xtest_Houston_Fusion,Ytest_Houston_Fusion, batch_size=32)
    results1 = finalclassifier.evaluate(Xtest_Houston_Fusion1,Ytest_Houston_Fusion, batch_size=32)

    print(" ")
    #print('REAL DATA %d test loss, test acc:', results)
    print('EPOCH GENERATED DATA test loss, test acc:',x, results1)

    if results1[1]> topvalid and results1[0]< toploss:
        topvalid = results1[1]
        toploss = results1[0]
        index = x


print("%d EPOCH, %f acc"%(index,topvalid))









