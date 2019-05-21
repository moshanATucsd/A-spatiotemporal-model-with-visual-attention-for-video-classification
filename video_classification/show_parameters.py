import argparse
import os
from models_rnn import ResearchModels


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument("--cnn", default="lenet", help="cnn model type", choices=["lenet", "stn", "dcn"])
args = parser.parse_args()

model_cnn = args.cnn
print("using cnn {0}...".format(model_cnn))

rm = ResearchModels(model=model_cnn)
print(rm.model.summary())

