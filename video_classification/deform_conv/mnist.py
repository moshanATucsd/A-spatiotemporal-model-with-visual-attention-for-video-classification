from __future__ import absolute_import, division

import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from PIL import Image
from scipy.misc import imresize
import numpy as np

# input image dimensions
image_size = (64, 64)
# image_size = (28, 28)


def resize_images(image_arrays, size=[32, 32]):

    resized_image_arrays = np.zeros((image_arrays.shape[0], size[0], size[1]), dtype='float32')
    for i, image_array in enumerate(image_arrays):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size=size, resample=Image.ANTIALIAS)

        resized_image_arrays[i,:,:] = np.asarray(resized_image)

    return resized_image_arrays

def get_mnist_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    # very important to subtract the mean
    mnist_mean = np.mean(X_train)
    mnist_std = np.std(X_train)

    X_train = X_train.astype('float32') - mnist_mean
    # X_train /= mnist_std
    X_test = X_test.astype('float32') - mnist_mean
    # X_test /= mnist_std

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    X_train = resize_images(X_train, image_size)
    X_test = resize_images(X_test, image_size)

    X_train = X_train[..., None]
    X_test = X_test[..., None]
    Y_train = keras.utils.to_categorical(y_train, 10)
    Y_test = keras.utils.to_categorical(y_test, 10)



    return (X_train, Y_train), (X_test, Y_test)


def get_gen(set_name, batch_size, translate, scale,
            shuffle=True):
    if set_name == 'train':
        (X, Y), _ = get_mnist_dataset()
    elif set_name == 'test':
        _, (X, Y) = get_mnist_dataset()

    image_gen = ImageDataGenerator(
        zoom_range=scale,
        width_shift_range=translate,
        height_shift_range=translate
    )
    gen = image_gen.flow(X, Y, batch_size=batch_size, shuffle=shuffle)
    return gen
