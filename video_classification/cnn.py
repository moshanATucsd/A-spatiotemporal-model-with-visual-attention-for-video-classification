import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

import keras.backend.tensorflow_backend as KTF

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np

import tensorflow as tf

#ignore warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.5):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

(trainData, trainLabels), (testData, testLabels) = cifar10.load_data()

num_classes = 10

trainLabels = keras.utils.to_categorical(trainLabels, num_classes)
testLabels = keras.utils.to_categorical(testLabels, num_classes)

#pre-processing
trainData = trainData.astype('float32')
testData = testData.astype('float32')
#very important to subtract the mean
trainData -= np.mean(trainData, axis=0, keepdims=True)
testData -= np.mean(testData, axis=0, keepdims=True)
trainData /= 255
testData /= 255

epochs = 25
batch_size = 128

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=trainData.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(BatchNormalization())
model.add(Activation('softmax'))

opt = SGD(lr=0.1, decay=0, momentum=0, nesterov=False)
#opt = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
#opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# print(model.summary())

# from keras.utils import plot_model
# plot_model(model, to_file='model.png')

hist = model.fit(trainData, trainLabels,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(testData, testLabels),
          shuffle=True)
# model.save_weights("cnn_weights.hdf5", overwrite=True)

# visualizing losses and accuracy

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']

ax = plt.figure().gca()
plt.plot(train_loss)
plt.plot(val_loss)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train_loss vs Val_loss')
plt.grid(True)
plt.legend(['Train_loss', 'Val_loss'])
plt.style.use(['classic'])

# ax = plt.figure().gca()
# plt.plot(train_acc)
# plt.plot(val_acc)
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Train_acc vs Val_acc')
# plt.grid(True)
# plt.legend(['Train_acc', 'Val_acc'], loc=4)
# plt.style.use(['classic'])

score = model.evaluate(testData, testLabels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.show()