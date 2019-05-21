"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import numpy as np
import os
from data_mnist import DataSet
from extractor import Extractor
from tqdm import tqdm
import keras.backend as K
import argparse
import matplotlib.pyplot as plt
from models_cnn import cnnModels
from keras.preprocessing import image
from keras.datasets import mnist

#export CUDA_VISIBLE_DEVICES='0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument("--deform", type=str, default="normal", help="deform type", choices=["normal", "rot", "scale", "rot_scale"])
parser.add_argument("--model", type=str, default="lenet", help="type of network", choices=["lenet", "stn"])
args = parser.parse_args()

deformation = args.deform
print("using deformation {0}...".format(deformation))

model_name = args.model
print("using model {0}...".format(model_name))

# Set defaults.
seq_length = 50
class_limit = None  # Number of classes to extract. Can be 1-101 or None for all.

# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit,deformation=deformation,model_cnn=model_name)

# model_name = "lenet"
# model_name = "stn"
# model_name = "dcn"
# weights_path = "models/lenet_weights.hdf5"
weights_path = "models/" + "{}".format(model_name) + ".hdf5"
# weights_path = "models/dcn.hdf5"
# get the model.
parts = weights_path.split('/')
filename = parts[1]
model = filename.split('.')[0]
# Get model with pretrained weights.
image_width = 64
base_model = cnnModels(model, image_shape=(image_width, image_width, 1))
base_model.model.load_weights(weights_path, by_name=True)

XX = base_model.model.input
YY = base_model.model.layers[0].output
F = K.function([XX], [YY])

directory = './visualization' + '{}'.format(model_name) + '_' + '{}'.format(deformation) + '/'

if not os.path.exists(directory):
    os.makedirs(directory)

(trainData, trainLabel), (testData, testLabel) = mnist.load_data()
trainData = trainData.astype('float32')
# very important to subtract the mean
mnist_mean = np.mean(trainData)
std = np.std(trainData)

def to_binary(img, lower, upper):
    return (lower < img) & (img < upper)

# Loop through data.
pbar = tqdm(total=len(data.data))
# count = 0
for video in data.data:

    # Get the path to the sequence for this video.
    # path = directory + video[2] + '-' + str(count) + '.png'

    # Get the frames for this video.
    frames = data.get_frames_for_sample(video)

    # Now downsample to just the ones we need.
    frames = data.rescale_list(frames, seq_length)

    # Now loop through and extract features to build the sequence.
    sequence = []
    count = 0
    for image_path in frames:
        count += 1
        # if count < 30:
        #     continue
        fig, (ax1, ax2) = plt.subplots(1, 2)
        print("image path ", image_path)
        parts = image_path.split('/')
        filename = parts[1]
        model = filename.split('.')[0]
        img = image.load_img(image_path, target_size=(image_width, image_width), grayscale=True)
        x = image.img_to_array(img)
        x -= mnist_mean
        # x /= std
        x /= 255
        x = np.expand_dims(x, axis=0)
        Xresult = F([x.astype('float32')])
        Xresult = np.squeeze(Xresult)
        Xresult *= 255
        Xresult += mnist_mean
        ax1.set_title('Original')
        ax1.imshow(img, cmap='gray')
        ax1.axis('off')
        ax2.set_title('STN')
        # plt.imshow(np.squeeze(Xresult), cmap='gray')
        # Xresult = to_binary(Xresult, 10, 255)
        ax2.imshow(Xresult, cmap='gray')
        ax2.axis('off')
        plt.show()
        raw_input("Press Enter to continue...")

    pbar.update(1)
    # count += 1

pbar.close()