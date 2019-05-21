from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping
import argparse

import tensorflow as tf

#lenet 98.58, 0.0418
#stn 99.21, 0.0265

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

from deform_conv.mnist import get_gen

import numpy as np

from models_cnn import cnnModels

#to suppress tensor flow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#scaled mnist
batch_size = 32
n_train = 60000
n_test = 10000

num_classes = 10
epochs = 5

steps_per_epoch = int(np.ceil(n_train / batch_size))
validation_steps = int(np.ceil(n_test / batch_size))

train_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=True
)
test_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 1.0), translate=0.0,
    shuffle=False
)
train_scaled_gen = get_gen(
    'train', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=True
)
test_scaled_gen = get_gen(
    'test', batch_size=batch_size,
    scale=(1.0, 2.5), translate=0.2,
    shuffle=False
)

parser = argparse.ArgumentParser()
parser.add_argument("--cnn", default="lenet", help="cnn model type", choices=["lenet", "stn", "dcn"])
args = parser.parse_args()

model_name = args.cnn
print("training ", model_name)
weights_path = "models/" + model_name + ".hdf5"
# Get model with pretrained weights.
img_width = 64
base_model = cnnModels(model_name, image_shape=(img_width, img_width, 1))

print(base_model.model.summary())

load_weights = False

#train or load weights
if load_weights:
    base_model.model.load_weights(weights_path, by_name=True)
else:
    # use early stopping to prevent overfitting
    # early_stopping = [EarlyStopping(monitor='acc', min_delta=0, patience=0, verbose=0, mode='auto')]
    # hist = base_model.model.fit(trainData, trainLabel,
    #                  batch_size=batch_size,
    #                  epochs=epochs,
    #                  verbose=1,
    #                  validation_data=(testData, testLabel),
    #                  callbacks=early_stopping,
    #                  validation_split=0.2)

    base_model.model.fit_generator(
        train_scaled_gen, steps_per_epoch=steps_per_epoch,
        epochs=epochs, verbose=1,
        validation_data=test_scaled_gen, validation_steps=validation_steps
    )

    base_model.model.save_weights(weights_path, overwrite=True)

