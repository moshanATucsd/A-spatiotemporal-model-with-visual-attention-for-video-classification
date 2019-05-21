from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
from models_cnn import cnnModels
from keras.datasets import mnist

class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model
        # model = 'dcn'
        parts = weights.split('/')
        filename = parts[1]
        model = filename.split('.')[0]
        # Get model with pretrained weights.
        base_model = cnnModels(model, image_shape=(64, 64, 1))

        base_model.model.load_weights(weights, by_name=True)

        # We'll extract features at the final pool layer.
        self.model = Model(
            inputs=base_model.model.input,
            outputs=base_model.model.get_layer('fc1').output
        )

        (trainData, trainLabel), (testData, testLabel) = mnist.load_data()
        trainData = trainData.astype('float32')
        # very important to subtract the mean
        self.mnist_mean = np.mean(trainData)
        self.std = np.std(trainData)

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(64, 64), grayscale=True)
        x = image.img_to_array(img)
        #z transform
        x -= self.mnist_mean
        # x /= self.std
        x /= 255
        x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features