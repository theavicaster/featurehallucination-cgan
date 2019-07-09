
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
os.environ["CUDA_VISIBLE_DEVICES"]="2"
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))


import keras
from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D,Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop,SGD, Adam
from functools import partial
from keras import layers, models
from keras.models import load_model

def loadIndianPinesData():

    data = sio.loadmat('/home/SharedData/Avinandan/IndianPines/Indian_pines_corrected.mat')['indian_pines_corrected']

    labels = sio.loadmat('/home/SharedData/Avinandan/IndianPines/Indian_pines_gt.mat')['indian_pines_gt']

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



data, indian_pines_gt = loadIndianPinesData()

data_mean = data.mean(axis=(0, 1), keepdims=True)
data_std = data.std(axis=(0, 1), keepdims=True)
data = (data - data_mean)/(data_std)
print(data.dtype)
data = (data - data.mean(axis=(0, 1), keepdims=True))/(data.std(axis=(0, 1), keepdims=True))
data = data.astype(np.float32)
print(data.dtype)

data, Y = createPatches(data, indian_pines_gt, windowSize=15)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y = onehot_encoder.fit_transform(integer_encoded)

#print(X.shape,Y.shape)

data1 = data[:,:,:,0:150]
data2 = data[:,:,:,150:200]

classifier = load_model('/home/SharedData/Avinandan/TeacherStudentExperiments/IndianPines/2N/Concatenated/BandSplitExperiments/75-25/twostream.h5')
streamone = Model(inputs=classifier.get_layer('Indian_Pines_Input_1').input,
                       outputs=classifier.get_layer('Indian_Pines_Output_1').output)
streamtwo = Model(inputs=classifier.get_layer('Indian_Pines_Input_2').input,
                       outputs=classifier.get_layer('Indian_Pines_Output_2').output)
#generator = load_model('/home/SharedData/Avinandan/Semi Supervised GAN/2N/generator2N.h5')
#discriminator = load_model('/home/SharedData/Avinandan/Semi Supervised GAN/2N/discriminator.h5')
        

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

random_latent_vectors = np.random.normal(0,1, size = (10249,latent_dim)) #sample random points

#print(random_latent_vectors.shape,data2.shape,data1.shape)


top = 0

#4900 5050
for x in range(3000,6000,10):

    generator = load_model('/home/SharedData/Avinandan/TeacherStudentExperiments/IndianPines/2N/Concatenated/BandSplitExperiments/75-25/models/generator2N-{0}.h5'.format(x))
    #generator = load_model('/home/SharedData/Avinandan/Semi Supervised GAN/2N/generator2N.h5')
                                             #latent, conditional
    generated_images = generator.predict([random_latent_vectors,data2])
    #generated_images = generator.predict([random_latent_vectors,data1],verbose=1)

    #generated_images = np.reshape(generated_images,(,100))
    #print(generated_images.shape)

    X_indian_pines1 = np.concatenate((generated_images,data2),axis=1)

    #print(X_indian_pines1.shape,Y.shape)

    from sklearn.model_selection import train_test_split


    #(Xtrain_indian_pines, Xtest_indian_pines,Ytrain_indian_pine,Ytest_indian_pines,) = train_test_split(X_indian_pines,Y_indian_pines, random_state = 666)

    #(Xtrain_indian_pines, Xtest_indian_pines,Ytrain_indian_pine,Ytest_indian_pines,) = train_test_split(data,Y, random_state = 666)

    (Xtrain_indian_pines1, Xtest_indian_pines1,Ytrain_indian_pine,Ytest_indian_pines,) = train_test_split(X_indian_pines1,Y, random_state = 666, test_size = 0.75)

    #results = onestream.evaluate([Xtest_indian_pines, Xtest_indian_pines],Ytest_indian_pines, batch_size=32)

    #print('test loss, test acc:', results)

    sgd = SGD(lr=0.0001, momentum=0.9, decay=1e-6)
    finalclassifier.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    #results = finalclassifier.evaluate(Xtest_indian_pines,Ytest_indian_pines, batch_size=32)
    results1 = finalclassifier.evaluate(Xtest_indian_pines1,Ytest_indian_pines, batch_size=32)

    print(" ")
    #print('REAL DATA %d test loss, test acc:', results)
    print('EPOCH %d GENERATED DATA %d test loss, test acc:',x, results1)

    if results1[1]> top:
        top = results1[1]
        index = x


print("%d EPOCH, %f acc",index,top)









