
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
config.gpu_options.per_process_gpu_memory_fraction = 0.5
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
   
    msdata = np.load('/home/SharedData/saurabh/HS/multispectralData.npy')
    pcdata = np.load('/home/SharedData/saurabh/HS/panchromaticData.npy')
    labels = np.load('/home/SharedData/saurabh/HS/labelsData.npy')
    
    return msdata, pcdata, labels






msdata, pcdata, gt = loadData()

msdata=np.transpose(msdata,(0,2,3,1))
pcdata=np.transpose(pcdata,(0,2,3,1))
gt=np.transpose(gt,(1,0))
gt=np.ravel(gt)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(gt)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y = onehot_encoder.fit_transform(
    integer_encoded)


#preprocessing b/w -1 and +1

msdata = msdata.astype(np.float32)
pcdata = pcdata.astype(np.float32)

it2 = 5000
for it1 in range(0,80000,5000):

    print(it1,it2)
    msdata[it1:it2,:,:,:] = (msdata[it1:it2,:,:,:] - msdata[it1:it2,:,:,:].mean(axis=(0, 1, 2), keepdims=True))/(msdata[it1:it2,:,:,:].std(axis=(0, 1,2), keepdims=True))
    pcdata[it1:it2,:,:,:] = (pcdata[it1:it2,:,:,:] - pcdata[it1:it2,:,:,:].mean(axis=(0,1, 2), keepdims=True))/(pcdata[it1:it2,:,:,:].std(axis=(0,1,2), keepdims=True))
    it2 += 5000

print(msdata.shape, pcdata.shape)

#from sklearn.model_selection import train_test_split

#(X_train_one, Xtest_one, Y, Ytest) = train_test_split(msdata, gt, random_state = 666, test_size= 0.75)
#(X_train_two, Xtest_two, Y, Ytest) = train_test_split(pcdata, gt, random_state = 666, test_size = 0.75 )

classifier = load_model('/home/SharedData/Avinandan/TeacherStudentExperiments/MultiModal/twostream.h5')
streamone = Model(inputs=classifier.get_layer('MultiModal_Input_1').input,
                       outputs=classifier.get_layer('MultiModal_Output_1').output)
streamtwo = Model(inputs=classifier.get_layer('Multimodal_Input_2').input,
                       outputs=classifier.get_layer('MultiModal_Output_2').output)
#generator = load_model('/home/SharedData/Avinandan/Semi Supervised GAN/2N/generator2N.h5')
#discriminator = load_model('/home/SharedData/Avinandan/Semi Supervised GAN/2N/discriminator.h5')

print(classifier.summary())
        

finalclassifierinput = Input(shape=(2*n_features,))

finalclassifier = finalclassifierinput

for layer in classifier.layers[35:]:
    finalclassifier = layer(finalclassifier)

# create the model
#new_model = Model(layer_input, x)

finalclassifier =  Model(inputs=finalclassifierinput,
                       outputs=finalclassifier)
print(finalclassifier.summary())


data1 = streamone.predict(msdata)
data2 = streamtwo.predict(pcdata)


data=np.concatenate((data1, data2), axis = 1)

random_latent_vectors = np.random.normal(0,1, size = (data1.shape[0],latent_dim)) #sample random points

#print(random_latent_vectors.shape,data2.shape,data1.shape)


topvalid = 0
toploss = 5000

#610
#2025 98.28
#2517 98.51

for x in range(3350,3450,1):

    generator = load_model('/home/SharedData/Avinandan/TeacherStudentExperiments/MultiModal/modelstwo/generator-{0}.h5'.format(x))
    #generator = load_model('/home/SharedData/Avinandan/Semi Supervised GAN/2N/generator2N.h5')
                                             #latent, conditional
    generated_images = generator.predict([random_latent_vectors,data1])
    #generated_images = generator.predict([random_latent_vectors,data1],verbose=1)

    #generated_images = np.reshape(generated_images,(,100))
    #print(generated_images.shape)

    X_MultiModal1 = np.concatenate((data1, generated_images),axis=1)

    #print(X_MultiModal1.shape,Y.shape)

    from sklearn.model_selection import train_test_split


    #(Xtrain_MultiModal, Xtest_MultiModal,Ytrain_MultiModal,Ytest_MultiModal,) = train_test_split(X_MultiModal,Y_MultiModal, random_state = 666)

    #(Xtrain_MultiModal, Xtest_MultiModal,Ytrain_MultiModal,Ytest_MultiModal,) = train_test_split(data,Y, random_state = 666)

    #(generated_images_train, generated_images_test,Ytrain_MultiModal,Ytest_MultiModal,) = train_test_split(generated_images,Y, random_state = 666)
    #(secondstream_train, secondstream_test,Ytrain_MultiModal,Ytest_MultiModal,) = train_test_split(data2,Y, random_state = 666)
    #(firststream_train, firststream_test,Ytrain_MultiModal,Ytest_MultiModal,) = train_test_split(data1,Y, random_state = 666)

    #results = onestream.evaluate([Xtest_MultiModal, Xtest_MultiModal],Ytest_MultiModal, batch_size=32)

    #print('test loss, test acc:', results)

    (Xtrain_MultiModal1, Xtest_MultiModal1,Ytrain_MultiModal,Ytest_MultiModal,) = train_test_split(X_MultiModal1,Y, random_state = 666, test_size = 0.75)

    sgd = SGD(lr=0.0001, momentum=0.9, decay=1e-6)
    finalclassifier.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    #results = finalclassifier.evaluate(Xtest_MultiModal,Ytest_MultiModal, batch_size=32)
    results1 = finalclassifier.evaluate(Xtest_MultiModal1,Ytest_MultiModal, batch_size=32)

    print(" ")
    #print('REAL DATA %d test loss, test acc:', results)
    print('EPOCH GENERATED DATA test loss, test acc:',x, results1)

    if results1[1]> topvalid and results1[0]< toploss:
        topvalid = results1[1]
        toploss = results1[0]
        index = x


print("%d EPOCH, %f acc",index,topvalid)









