from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
import datetime
import numpy as np
from data_loader import DataLoader
from sklearn.metrics import accuracy_score
from keras.models import load_model


class ConvolutionalNeuralNetworks():
    def __init__(self):
        # Input shape
        self.img_rows = 431
        self.img_cols = 1162
        self.channels = 3
        #self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_shape = (20,11,1)

        # Configure data loader
        self.data_loader = DataLoader( img_res=(self.img_rows, self.img_cols))
        self.data_loader.reshape()

        # Build the network
        optimizer = Adam(lr=0.004, beta_1=0.9, beta_2=0.999)
        self.CNN_Network = self.build_CNN_Network()
        self.CNN_Network.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        # Input images
        Xtr = Input(shape=self.img_shape)

        # Output labels
        Ypred = self.CNN_Network(Xtr)

    def build_CNN_Network(self):

        def conv2d(layer_input, filters, f_size=4, stride=2, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=stride, padding='valid')(layer_input)
            #d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def maxpooling2d(layer_input, f_size, stride=2):
            d = MaxPooling2D(pool_size=f_size, strides=stride, padding='valid')(layer_input)
            return d

        def flatten(layer_input):
            d = Flatten()(layer_input)
            return d

        def dense(layer_input, f_size, dr=True, lastLayer=True):
            if lastLayer:
                d = Dense(f_size, activation='softmax')(layer_input)
            else:
                d = Dense(f_size, activation='relu')(layer_input)
                #d = LeakyReLU(alpha=0.2)(d)
                if dr:
                    d = Dropout(0.5)(d)
            return d

        # LeNet-5 layers
        d0 = Input(shape=self.img_shape) # Image input
        d1 = conv2d(d0, filters=32, f_size=2, stride=1, bn=True)
        d2 = maxpooling2d(d1, f_size=2, stride=2)
        d3 = Dropout(0.25)(d2)
        d4 = flatten(d3)
        d5 = dense(d4, f_size=128, dr=True, lastLayer=False)
        d6 = dense(d5, f_size=5, dr=False, lastLayer=True)


        '''
        d3 = conv2d(d2, filters=16, f_size=5, stride=1, bn=True)
        d4 = maxpooling2d(d3, f_size=2, stride=2)
        d5 = flatten(d4)
        d6 = dense(d5, f_size=120, dr=True, lastLayer=False)
        d7 = dense(d6, f_size=84, dr=True, lastLayer=False)
        d8 = dense(d7, f_size=10, dr=True, lastLayer=False)
        d9 = dense(d8, f_size=5, dr=False, lastLayer=True) #add
        '''

        return Model(d0, d7)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            for batch_i, (Xtr, labels) in enumerate(self.data_loader.load_batch(batch_size)):
                #  Training
                crossentropy_loss = self.CNN_Network.train_on_batch(Xtr, labels)

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [Training loss: %f, Training acc: %3d%%] time: %s" % (
                    epoch + 1, epochs,
                    batch_i + 1, self.data_loader.n_batches - 1,
                    crossentropy_loss[0], 100 * crossentropy_loss[1],
                    elapsed_time))
                # If at save interval => do validation and save model
                if (batch_i + 1) % sample_interval == 0:
                    self.validation(epoch)

    def validation(self, epoch):
        Xte, Xte_labels = self.data_loader.load_data(batch_size=8)
        pred_labels = self.CNN_Network.predict(Xte)
        print("Validation acc: " + str(
            int(accuracy_score(np.argmax(Xte_labels, axis=1), np.argmax(pred_labels, axis=1)) * 100)) + "%")
        self.CNN_Network.save('./saved_model/CNN_Network_on_epoch_%d.h5' % epoch)


if __name__ == '__main__':
    # # training model
    my_CNN_Model = ConvolutionalNeuralNetworks()
    my_CNN_Model.train(epochs=100, batch_size=2, sample_interval=397)



