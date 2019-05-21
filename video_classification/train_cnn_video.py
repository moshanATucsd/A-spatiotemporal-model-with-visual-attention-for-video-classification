from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
from keras.callbacks import EarlyStopping
from deform_conv.mnist import get_gen
from keras.optimizers import Adam, RMSprop, SGD, Adadelta

import time

from spatial_transformer import SpatialTransformer


import numpy as np

from models_cnn import cnnModels
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--deform", type=str, default="normal", help="deform type", choices=["normal", "rot", "scale", "rot_scale"])
args = parser.parse_args()

deformation = args.deform
print("using deformation {0}...".format(deformation))

#to suppress tensor flow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

batch_size = 20

epochs = 100
seq_length = 50
class_limit = None
image_shape = (64, 64, 1)
num_classes = 100

# initial weights
b = np.zeros((2, 3), dtype='float32')
b[0, 0] = 1
b[1, 1] = 1
W = np.zeros((50, 6), dtype='float32')
weights = [W, b.flatten()]

locnet = Sequential()
locnet.add(MaxPooling2D(pool_size=(2, 2), input_shape=image_shape))
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
                             output_size=(30, 30), input_shape=image_shape))
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
model.add(Dense(num_classes, activation='softmax', name='predictions'))

# optimizer = keras.optimizers.Adadelta()
optimizer = SGD(lr=1e-4, decay=1e-6, momentum=0.5, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=optimizer,
                   metrics=['accuracy'])

from data_mnist import DataSet

def get_generators():

    data = DataSet(
        seq_length=seq_length,
        class_limit=class_limit,
        image_shape=image_shape,
    )

    # print("class ", data.classes)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_path = '/data3/moshan/cogs260_class_project/video_classification/data/moving_mnist/train2'
    train_path = train_path + '_' + '{}'.format(deformation) + '/'
    test_path = '/data3/moshan/cogs260_class_project/video_classification/data/moving_mnist/test2'
    test_path = test_path + '_' + '{}'.format(deformation) + '/'

    train_generator = train_datagen.flow_from_directory(
        train_path,
        color_mode="grayscale",
        target_size=(64, 64),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        test_path,
        color_mode = "grayscale",
        target_size=(64, 64),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')

    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    return train_generator, validation_generator, steps_per_epoch

# model_name = 'lenet'
model_name = 'stn'
print("video: training ", model_name)
weights_path = "models_video/" + model_name + "_" + "{}".format(deformation) + ".hdf5"
# base_model = cnnModels(model_name, image_shape=(64, 64, 1))

# print(base_model.model.summary())

load_weights = False


train_generator, validation_generator, steps_per_epoch = get_generators()

# Helper: TensorBoard
tb = TensorBoard(log_dir='./cnn_video/logs/'+'{}'.format(deformation))

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# # Helper: Save results.
# timestamp = time.time()
# csv_logger = CSVLogger('./cnn_video/logs/' + model_name + '-' + 'training-' + \
#                        str(timestamp) + '.log')

model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=10,
    callbacks=[tb, early_stopper],
    epochs=epochs)

model.save_weights(weights_path, overwrite=True)


