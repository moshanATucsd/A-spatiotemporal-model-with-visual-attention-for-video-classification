"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD, Adadelta
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
import sys
from spatial_transformer import SpatialTransformer
import numpy as np

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048):
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
        self.dropout = 0

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        # if self.nb_classes >= 10:
        #     metrics.append('top_k_categorical_accuracy')

        # print("model name ", model)
        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        elif model == 'gru':
            print("Loading GRU model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.gru()
        elif model == 'cnn':
            print("Loading CNN model.")
            self.input_shape = features_length * seq_length
            self.model = self.cnn()
        elif model == 'mlp':
            print("Loading simple MLP.")
            self.input_shape = features_length * seq_length
            self.model = self.mlp()
        elif model == 'conv_3d':
            print("Loading Conv3D")
            self.input_shape = (seq_length, 64, 64, 1)
            self.model = self.conv_3d()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        # optimizer = Adam(lr=1e-3)  # aggressively small learning rate
        optimizer = RMSprop(lr=1e-3)
        # optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.5, nesterov=True)
        # optimizer = Adadelta()
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(8, return_sequences=True, input_shape=self.input_shape,
                       dropout=self.dropout))
        model.add(Flatten())
        # model.add(Dense(64, activation='relu'))
        # model.add(Dropout(self.dropout))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def gru(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        # model.add(LSTM(4, return_sequences=True, input_shape=self.input_shape,
        #                dropout=self.dropout))
        model.add(GRU(32, return_sequences=True, input_shape=self.input_shape,
                       dropout=self.dropout))
        model.add(Flatten())
        # model.add(Dense(64, activation='relu'))
        # model.add(Dropout(self.dropout))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def cnn(self):

        model = Sequential()
        model.add(Dense(self.nb_classes, activation='softmax', input_dim=self.input_shape))

        return model

    def mlp(self):
        """Build a simple MLP."""
        # Model.
        model = Sequential()
        model.add(Dense(256, input_dim=self.input_shape))
        model.add(Dropout(self.dropout))
        # model.add(Dense(512))
        # model.add(Dropout(self.dropout))
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