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

import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument("--deform", type=str, default="normal", help="deform type", choices=["normal", "rot", "scale", "rot_scale"])
parser.add_argument("--model", type=str, default="lenet", help="type of network", choices=["lenet", "stn", "dcn"])
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
model = Extractor(weights_path)

directory = './data/sequences_' + '{}'.format(model_name) + '_' + '{}'.format(deformation) + '/'

if not os.path.exists(directory):
    os.makedirs(directory)

# Loop through data.
pbar = tqdm(total=len(data.data))
for video in data.data:

    # Get the path to the sequence for this video.
    path = directory + video[2] + '-' + str(seq_length) + \
        '-features.txt'

    # # Check if we already have it.
    # if os.path.isfile(path):
    #     pbar.update(1)
    #     continue

    # Get the frames for this video.
    frames = data.get_frames_for_sample(video)

    # Now downsample to just the ones we need.
    frames = data.rescale_list(frames, seq_length)

    # Now loop through and extract features to build the sequence.
    sequence = []
    for image in frames:
        features = model.extract(image)
        sequence.append(features)

    # Save the sequence.
    np.savetxt(path, sequence)

    pbar.update(1)

pbar.close()