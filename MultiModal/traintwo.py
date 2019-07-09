from __future__ import print_function, division

import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras import layers, models
from keras.models import load_model




import scipy.io as sio

import matplotlib.pyplot as plt

import sys

import numpy as np
np.random.seed(666)


import os
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

def loadData():
   
    msdata = np.load('/home/SharedData/saurabh/HS/multispectralData.npy')
    pcdata = np.load('/home/SharedData/saurabh/HS/panchromaticData.npy')
    labels = np.load('/home/SharedData/saurabh/HS/labelsData.npy')
    
    return msdata, pcdata, labels





class GAN():
    def __init__(self):
        #self.img_rows = 145
        #self.img_cols = 145
        #self.channels = 100
        #self.img_shape = (self.channels,)
        self.latent_dim = 100
        self.n_features = 200

        


        optimizerdisc = SGD(0.001, clipnorm = 1. , clipvalue = 0.5)
        optimizergen = Adam(0.0002,0.5,clipnorm = 1. , clipvalue = 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        #self.discriminator = load_model('/home/SharedData/Avinandan/Semi Supervised GAN/Average/models/discriminator2N-5999.h5')

        self.discriminator.compile(loss='categorical_crossentropy',
            optimizer=optimizerdisc,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        #self.generator = load_model('/home/SharedData/Avinandan/Semi Supervised GAN/Average/models/generator2N-5999.h5')


        # Build the feature extractors
        self.classifier = load_model('/home/SharedData/Avinandan/TeacherStudentExperiments/MultiModal/twostream.h5')
        self.streamone = Model(inputs=self.classifier.get_layer('MultiModal_Input_1').input,
                               outputs=self.classifier.get_layer('MultiModal_Output_1').output)
        self.streamtwo = Model(inputs=self.classifier.get_layer('Multimodal_Input_2').input,
                               outputs=self.classifier.get_layer('MultiModal_Output_2').output)

        # The generator takes noise and conditional features as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        conditional_input= Input(shape=(self.n_features,))
        img = self.generator([z, conditional_input])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([z,conditional_input], validity)
        self.combined.compile(loss='categorical_crossentropy', optimizer=optimizergen, metrics =['accuracy'])


    def build_generator(self):

        #model = Sequential()

        #noise to generator and input features of one channel
        generator_input = Input(shape=(self.latent_dim ,)) 
        conditional_input= Input(shape=(self.n_features,))
        
        #model.add(Dense(256, input_dim=self.latent_dim))

        model= layers.Concatenate()([generator_input, conditional_input])

        model = layers.Dense(256)(model)
        model = layers.LeakyReLU(alpha=0.2)(model)
        model = layers.BatchNormalization(momentum=0.8)(model)

        model = layers.Dense(512)(model)
        model = layers.LeakyReLU(alpha=0.2)(model)
        model = layers.BatchNormalization(momentum=0.8)(model)

        model = layers.Dense(1024)(model)
        model = layers.LeakyReLU(alpha=0.2)(model)
        model = layers.BatchNormalization(momentum=0.8)(model)

        model = layers.Dense(np.prod((self.n_features,)), activation='tanh')(model)
        model = layers.Reshape((self.n_features,))(model)
        
        genmodel = models.Model(inputs=[generator_input,conditional_input],outputs =  model)
        #print(genmodel.summary())

        img = genmodel([generator_input,conditional_input])
       
        #what to return????
        return Model([generator_input,conditional_input], img)
        #have we ever really returned

    def build_discriminator(self):

        model = Sequential()

        #model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(1024 , input_shape = (self.n_features,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dense(1, activation='sigmoid'))
        #model.add(Lambda(lambda x: x / self.temp))
        model.add(Dense(16, activation='softmax', name='Classifier_Output'))
        #model.summary()

        img = Input(shape=(self.n_features,))
        validity = model(img)

        return Model(img, validity)

  

    def train(self, epochs, batch_size=128):



        msdata, pcdata, gt = loadData()

        msdata=np.transpose(msdata,(0,2,3,1))
        pcdata=np.transpose(pcdata,(0,2,3,1))
        gt=np.transpose(gt,(1,0))
        gt=np.ravel(gt)


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

        from sklearn.model_selection import train_test_split

        (X_train_one, Xtest_one, Y, Ytest) = train_test_split(msdata, gt, random_state = 666, test_size= 0.75)
        (X_train_two, Xtest_two, Y, Ytest) = train_test_split(pcdata, gt, random_state = 666, test_size = 0.75 )

        X_train_one = self.streamone.predict(X_train_one)
        X_train_two = self.streamtwo.predict(X_train_two)

        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import OneHotEncoder

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(Y)

        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        Y = onehot_encoder.fit_transform(
            integer_encoded)


        prepend = np.zeros((Y.shape[0],Y.shape[1]))
        #print(prepend.size)
        Y_fake = np.concatenate((prepend,Y),axis=1)
        Y_real = np.concatenate((Y,prepend),axis=1)
        print(Y.shape,Y_real.shape,Y_fake.shape)



        #print(X_train_one.shape, "EXTRACTED FEATURE SHAPE")


        #Adversarial ground truths
        #valid = np.ones((batch_size, 1))
        #fake = np.zeros((batch_size, 1))
        #fake = np.zeros((batch_size,Y.shape[1]))
        #fake[:,0] = 1


        #print(fake)

       
        #label smoothing
        #valid = np.random.uniform(0.7,1.2,size=(batch_size,1))
        #fake = np.random.uniform(0.0,0.3,size=(batch_size,1))

        #print(valid.shape, Y.shape)

        for epoch in range(epochs):



			# ---------------------
			#  Train Discriminator
			# ---------------------

            self.discriminator.trainable = True
            self.generator.trainable = False

			# Select a random batch of images
            idx = np.random.randint(0, X_train_two.shape[0], batch_size)
            imgs = X_train_one[idx]
            valid_imgs = X_train_two[idx]
            valid = Y_real[idx]
            fake = Y_fake[idx]
            #valid[valid == 1] = np.random.uniform(0.8,1.0)
            for x in range(0, valid.shape[0]):
                for y in range(0, valid.shape[1]):
                    if valid[x, y] ==1:
                        valid[x, y] = np.random.uniform(0.8,1.2)
                    if fake[x, y] ==1:
                        fake[x, y] = np.random.uniform(0.8,1.2)
            
            #print(valid)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict([noise,imgs])
            #print(gen_imgs.shape)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(valid_imgs, valid)
            #print("[C acc.: %.2f%%]" % (100*d_loss_real[1]))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            for g in range(5):

                # ---------------------
                #  Train Generator
                # ---------------------

                self.discriminator.trainable=False
                self.generator.trainable=True

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))


                #Every 50 epochs, flips the labels for real labels
                if epoch%50 == 0 and epoch < epochs/2:
                    swapvar = valid[:,:]
                    valid[:,:] = fake[:,:]
                    fake[:,:] = swapvar[:,:]

                if epoch >= epochs/2:
                    self.generator.save('/home/SharedData/Avinandan/TeacherStudentExperiments/MultiModal/modelstwo/generator-{0}.h5'.format(epoch))
                    self.discriminator.save('/home/SharedData/Avinandan/TeacherStudentExperiments/MultiModal/modelstwo/discriminator-{0}.h5'.format(epoch))







    			# Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch([noise,imgs],valid)
                g_loss = np.asarray(g_loss)
                #print(g_loss.shape)
                

                #print(g_loss.shape)
                #print(d_loss.shape)ccc

    			# Plot the progress
                print ("%d [D loss: %f, real, fake, overall acc.: %.2f%%, %.2f%%, %.2f%%] [G loss: %f, acc.: %.2f%%]" % (epoch, d_loss[0], 100*d_loss_real[1], 100*d_loss_fake[1], 100*d_loss[1], g_loss[0], 100*g_loss[1] ))


                #plt.scatter(epoch,d_loss[0], s=5.)
                #plt.scatter(epoch,g_loss[0], s=5.)
                #plt.title('GAN Loss')
                #plt.ylabel('Loss')
                #plt.xlabel('Epoch')
                #plt.savefig('/home/SharedData/Avinandan/Semi Supervised GAN/ganloss.png')





if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=6000, batch_size=1024)

    #gan.generator.save('/home/SharedData/Avinandan/Semi Supervised GAN/2N/generator2N.h5')
    #gan.discriminator.save('/home/SharedData/Avinandan/Semi Supervised GAN/2N/discriminator2N.h5')
