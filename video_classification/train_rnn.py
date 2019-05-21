"""
Train our RNN on bottlecap or prediction files generated from our CNN.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models_rnn import ResearchModels
from data_mnist import DataSet
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import pickle

import argparse

def save(obj, name):
    try:
        filename = open(name + ".pickle","wb")
        pickle.dump(obj, filename)
        filename.close()
        return(True)
    except:
        return(False)

def train(data_type, seq_length, model, saved_model=None,
          concat=False, class_limit=None, image_shape=None,
          load_to_memory=False, deformation=None,model_cnn=None):
    # Set variables.
    nb_epoch = 10
    batch_size = 50

    validation_steps = 20

    # model_name = "lenet"
    model_name = model_cnn
    print("training rnn with....", model_name)

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath='./data/checkpoints/' + model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir='./data/logs')

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger('./data/logs/' + model + '-' + 'training-' + \
        str(timestamp) + '.log')

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            deformation=deformation,
            model_cnn=model_cnn
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape,
            deformation=deformation,
            model_cnn=model_cnn
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory(batch_size, 'train', data_type, concat)
        X_test, y_test = data.get_all_sequences_in_memory(batch_size, 'test', data_type, concat)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type, concat)
        val_generator = data.frame_generator(batch_size, 'test', data_type, concat)

    train_rounds = 300
    avg_acc = 0
    avg_loss = 0
    for _ in range(0, train_rounds):

        # Get the model.
        feature_length = 500
        rm = ResearchModels(len(data.classes), model, seq_length, saved_model, feature_length)

        # Fit!
        if load_to_memory:
            # Use standard fit.
            # hist = rm.model.fit(
            #     X,
            #     y,
            #     batch_size=batch_size,
            #     validation_data=(X_test, y_test),
            #     verbose=0,
            #     callbacks=[checkpointer, tb, early_stopper, csv_logger],
            #     epochs=nb_epoch)
            hist = rm.model.fit(
                X,
                y,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=0,
                callbacks=[early_stopper],
                epochs=nb_epoch)
        else:
            # Use fit generator.
            hist = rm.model.fit_generator(
                generator=generator,
                steps_per_epoch=steps_per_epoch,
                epochs=nb_epoch,
                verbose=1,
                callbacks=[checkpointer, tb, early_stopper, csv_logger],
                validation_data=val_generator,
                validation_steps=validation_steps)

        loss, acc = rm.model.evaluate(X_test, y_test, verbose=0)
        avg_acc += acc
        avg_loss += loss

    avg_acc /= train_rounds
    avg_loss /= train_rounds
    print('\nAvg test loss: {}, acc: {}\n'.format(avg_loss, avg_acc))

    text_file = open("Output.txt", "a")
    text_file.write("using cnn {0}...".format(model_cnn))
    text_file.write("using deformation {0}...".format(deformation))
    text_file.write('Avg test loss: {}, acc: {}\n'.format(avg_loss, avg_acc))
    text_file.close()

    # total_step = 20
    # avg_loss = 0
    # avg_acc = 0
    # for _ in range(0, total_step):
    #     loss, acc = rm.model.evaluate(X_test, y_test, verbose=0)
    #     avg_loss += loss
    #     avg_acc += acc
    #
    # avg_loss /= total_step
    # avg_acc /= total_step
    # print('\nAvg test loss: {}, acc: {}\n'.format(avg_loss, avg_acc))

    # #plot training
    # train_loss = hist.history['loss']
    # val_loss = hist.history['val_loss']
    # train_acc = hist.history['acc']
    # val_acc = hist.history['val_acc']
    #
    # filepath = model_name + "_hist"
    # save(hist.history, filepath)
    #
    # ax = plt.figure().gca()
    # plt.plot(train_loss)
    # plt.plot(val_loss)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Train_loss vs Val_loss')
    # plt.grid(True)
    # plt.legend(['Train_loss', 'Val_loss'])
    # plt.style.use(['classic'])
    #
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

    # plt.show()

    # score = rm.model.evaluate(X_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

import os

def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    parser = argparse.ArgumentParser()
    parser.add_argument("--deform", default="normal", help="deform type", choices=["normal", "rot", "scale", "rot_scale"])
    parser.add_argument("--cnn", default="lenet", help="cnn model type", choices=["lenet", "stn", "dcn"])
    parser.add_argument("--rnn", default="lstm", help="rnn model type", choices=["lstm", "gru"])
    args = parser.parse_args()

    deformation = args.deform
    print("using deformation {0}...".format(deformation))

    model_cnn = args.cnn
    print("using cnn {0}...".format(model_cnn))

    model_rnn = args.rnn
    print("using rnn {0}...".format(model_rnn))

    """These are the main training settings. Set each before running
    this file."""
    # model_rnn = 'gru'  # see `models.py` for more
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 50
    load_to_memory = True  # pre-load the sequences into memory

    # Chose images or features and image shape based on network.
    if model_rnn == 'conv_3d' or model_rnn == 'crnn':
        data_type = 'images'
        image_shape = (64, 64, 1)
        load_to_memory = False
    else:
        data_type = 'features'
        image_shape = None

    # MLP requires flattened features.
    if model_rnn == 'mlp' or model_rnn == 'cnn':
        concat = True
    else:
        concat = False

    train(data_type, seq_length, model_rnn, saved_model=saved_model,
          class_limit=class_limit, concat=concat, image_shape=image_shape,
          load_to_memory=load_to_memory, deformation=deformation,model_cnn=model_cnn)

if __name__ == '__main__':
    main()