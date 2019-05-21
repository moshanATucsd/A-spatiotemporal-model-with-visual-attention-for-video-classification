"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAvgPool2D, Activation, ZeroPadding2D
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
import sys

from deform_conv.layers import ConvOffset2D

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, image_shape=None, features_length=2048):
        """
        `model` = one of:
            lstm
            crnn
            mlp
            conv_3d
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()
        self.image_shape = image_shape

        h, w, c = image_shape

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'crnn_stn':
            print("Loading crnn_stn model.")
            self.input_shape = (seq_length, h, w, c)
            self.model = self.crnn_stn()
        elif model == 'crnn_dcn':
            print("Loading crnn_dcn model.")
            self.input_shape = (seq_length, h, w, c)
            self.model = self.crnn_dcn()
        elif model == 'baseline':
            print("Loading baseline.")
            self.input_shape = (seq_length, h, w, c)
            self.model = self.baseline()
        elif model == 'conv_3d':
            print("Loading Conv3D")
            self.input_shape = (seq_length, h, w, c)
            self.model = self.conv_3d()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=1e-3)  # aggressively small learning rate
        # optimizer = RMSprop(lr=0.01)
        # optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

    def crnn_stn(self):

        from spatial_transformer import SpatialTransformer
        import numpy as np

        # initial weights
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        W = np.zeros((20, 6), dtype='float32')
        weights = [W, b.flatten()]

        locnet = Sequential()
        locnet.add(MaxPooling2D(pool_size=(2, 2), input_shape=self.image_shape))
        locnet.add(Conv2D(20, (5, 5)))
        locnet.add(MaxPooling2D(pool_size=(2, 2)))
        locnet.add(Conv2D(20, (5, 5)))
        locnet.add(Dense(6, weights=weights))

        #vgg 16
        model = Sequential()
        model.add(TimeDistributed(SpatialTransformer(localization_net=locnet,
                output_size=(30, 30)), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(8, (3, 3),
            activation='relu', padding='same'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(32, (3, 3),
            activation='relu', padding='same')))

        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(16, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def crnn_dcn(self):
        #vgg 16
        model = Sequential()

        model.add(TimeDistributed(Conv2D(8, (3, 3),
            activation='relu', padding='same'), input_shape=self.input_shape))
        model.add(TimeDistributed(ConvOffset2D(8)))
        model.add(TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same')))
        model.add(TimeDistributed(ConvOffset2D(8)))

        model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same')))
        model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same')))

        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(500, return_sequences=True, input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Flatten())
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def baseline(self):

        model = Sequential()
        model.add(TimeDistributed(Conv2D(20, kernel_size=(5, 5),
                 padding="same",
                 activation='relu'),
                 input_shape=self.input_shape))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))
        model.add(TimeDistributed(Conv2D(50, kernel_size=(5, 5),
                         padding="same",
                         activation='relu')))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))
        model.add(TimeDistributed(Flatten()))
        model.add(TimeDistributed(Dense(500, activation='relu', name='fc1')))
        model.add(LSTM(500, return_sequences=True, input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Flatten())
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def conv_3d(self):
        """
        Build a 3D convolutional network, based loosely on C3D.
            https://arxiv.org/pdf/1412.0767.pdf
        """
        # Model.
        model = Sequential()
        model.add(Conv3D(
            32, (7,7,7), activation='relu', input_shape=self.input_shape
        ))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(64, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(128, (2,2,2), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
