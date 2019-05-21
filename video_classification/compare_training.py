import pickle

name = "lenet_hist"

filename = open(name + ".pickle", "rb")
obj = pickle.load(filename)
filename.close()