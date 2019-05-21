import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
from keras.callbacks import EarlyStopping

#dcn 102,483,280

from deform_conv.layers import ConvOffset2D
from spatial_transformer import SpatialTransformer

import numpy as np

class cnnModels():
    def __init__(self, model=None, image_shape=None):

        self.input_shape = image_shape
        self.num_classes = 10

        if model == 'lenet':
            print("Loading lenet model.")
            self.model = self.lenet()
        elif model == 'dcn':
            print("Loading dcn model.")
            self.model = self.dcn()
        elif model == 'stn':
            print("Loading stn model.")
            self.model = self.stn()

        optimizer = keras.optimizers.Adadelta()
        # optimizer = keras.optimizers.SGD(lr=1e-1, decay=1e-6, momentum=0.5, nesterov=True)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])

    def lenet(self):
        # lenet
        model = Sequential()
        # conv1
        model.add(Conv2D(20, kernel_size=(5, 5),
                         padding="same",
                         activation='relu',
                         input_shape=self.input_shape))
        # pool1
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # conv2
        model.add(Conv2D(50, kernel_size=(5, 5),
                         padding="same",
                         activation='relu'))
        # pool2
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        # ip1
        model.add(Dense(500, activation='relu', name='fc1'))
        # model.add(Dropout(0.5))
        # ip2
        model.add(Dense(self.num_classes, activation='softmax', name='predictions'))

        return model

    def dcn(self):
        # dcn
        model = Sequential()
        # conv1
        model.add(Conv2D(20, kernel_size=(5, 5),
                         padding="same",
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(ConvOffset2D(20))
        # pool1
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # conv2
        model.add(Conv2D(50, kernel_size=(5, 5),
                         padding="same",
                         activation='relu'))
        model.add(ConvOffset2D(50))
        # pool2
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        # ip1
        model.add(Dense(500, activation='relu', name='fc1'))
        # model.add(Dropout(0.5))
        # ip2
        model.add(Dense(self.num_classes, activation='softmax', name='predictions'))

        return model

    def stn(self):

        # initial weights
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        W = np.zeros((50, 6), dtype='float32')
        weights = [W, b.flatten()]

        locnet = Sequential()
        locnet.add(MaxPooling2D(pool_size=(2, 2), input_shape=self.input_shape))
        locnet.add(Conv2D(20, (5, 5)))
        locnet.add(MaxPooling2D(pool_size=(2, 2)))
        locnet.add(Conv2D(20, (5, 5)))

        locnet.add(Flatten())
        locnet.add(Dense(50))
        locnet.add(Activation('relu'))
        locnet.add(Dense(6, weights=weights))

        # stn
        model = Sequential()
        # conv1
        model.add(SpatialTransformer(localization_net=locnet,
            output_size=(32, 32), input_shape=self.input_shape))
        # model.add(SpatialTransformer(localization_net=locnet,
        #     output_size=self.input_shape, input_shape=self.input_shape))
        model.add(Conv2D(20, kernel_size=(5, 5),
                         padding="same",
                         activation='relu'))
        # pool1
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # conv2
        model.add(Conv2D(50, kernel_size=(5, 5),
                         padding="same",
                         activation='relu'))
        # pool2
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        # ip1
        model.add(Dense(500, activation='relu', name='fc1'))
        # model.add(Dropout(0.5))
        # ip2
        model.add(Dense(self.num_classes, activation='softmax', name='predictions'))

        return model