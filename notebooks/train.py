# coding: utf-8
import os

from keras.callbacks import EarlyStopping
from keras.layers import (AveragePooling2D, Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array

from keras.utils.vis_utils import plot_model

from sklearn.metrics import mean_squared_error

from PIL import ImageFile

from imageaesthetics import pipeline
from imageaesthetics import model
from imageaesthetics import losses
from imageaesthetics import utils

import pandas as pd

import argparse
import time

if __name__ == "__main__":

    start = time.time()

    parser = argparse.ArgumentParser(
        description='Triggers a teaching cycle')
    parser.add_argument('-d', '--description')
    parser.add_argument('-t', '--target')
    parser.add_argument('-n', '--number_of_samples_per_bucket')
    parser.add_argument('-b', '--base_model')
    parser.add_argument('-o', '--dropout_rate')
    parser.add_argument('-ld', '--learning_rate_dense')
    parser.add_argument('-la', '--learning_rate_all')
    parser.add_argument('-ed', '--epochs_dense')
    parser.add_argument('-ea', '--epochs_all')
    parser.add_argument('-dd', '--decay_dense')
    parser.add_argument('-da', '--decay_all')

    args = parser.parse_args()

    model_target = args.target
    model_description = args.description
    base_model = args.base_model

    learning_rate_dense = 1e-3
    if args.learning_rate_dense != None:
        learning_rate_dense = float(args.learning_rate_dense)

    learning_rate_all = 1e-3
    if args.learning_rate_all != None:
        learning_rate_all = float(args.learning_rate_all)

    epochs_dense = 10
    if args.epochs_dense != None:
        epochs_dense = float(args.epochs_dense)

    epochs_all = 20
    if args.epochs_all != None:
        epochs_all = float(args.epochs_all)

    decay_dense = 0.0
    if args.decay_dense != None:
        decay_dense = float(args.decay_dense)

    decay_all = 0.0
    if args.decay_all != None:
        decay_all = float(args.decay_all)

    dropout = 0.75
    if args.dropout_rate != None:
        dropout = float(args.dropout_rate)

    number_of_samples_per_bucket = None
    if args.number_of_samples_per_bucket != None:
        number_of_samples_per_bucket = int(args.number_of_samples_per_bucket)

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Constants
    DATA_FOLDER = "../data"
    IMAGE_PATH = "{}/ava_downloader/AVA_dataset/images".format(DATA_FOLDER)
    META_DATA_PATH = "{}/ava_downloader/AVA_dataset/".format(
        DATA_FOLDER)

    BATCH_SIZE = 32

    # Iteration specific
    FORCE_TRAIN_BOTTLENECK = False
    FORCE_TEST_BOTTLENECK = False
    FORCE_MODEL_FIT = False

    TARGET_SIZE = (224, 224)
    INPUT_SHAPE = TARGET_SIZE + (3,)
    N_CLASSES = 10

    if not os.path.exists(model_target):
        os.makedirs(model_target)

    pipeline.save_desc(model_target, model_description)

    # Load dataset
    print("\nloading images...")
    pipeline.load_data_set(DATA_FOLDER)

    # Split training, validation and test set
    print("\nsplitting dataset into train, validation and test set....")

    df, test_df = pipeline.load_df(model_target, META_DATA_PATH)

    df['image.id'] = df['image.id'].astype(str)
    test_df['image.id'] = test_df['image.id'].astype(str)
    if number_of_samples_per_bucket != None:
        df = df.groupby('rating.mean.bucket').head(
            number_of_samples_per_bucket).reset_index()

    train_df, validation_df = pipeline.split_data(model_target, df)

    # Model creation
    print("\ncreating model...")
    top_model = Sequential()
    top_model.add(Dropout(dropout))
    top_model.add(Dense(10, activation='softmax'))

    model = model.AestheticsModel(
        base_model_name=base_model, top_model=top_model,
        input_shape=INPUT_SHAPE, optimizer=Adam(lr=learning_rate_dense, decay=decay_dense), loss=losses.earth_mover_loss)

    # Prepare Image Generators
    print("\nprepare image generators...")

    train_generator, validation_generator = pipeline.get_image_generators(IMAGE_PATH, utils.to_dict(
        train_df,  'image.id', ['1', '2', '3',
                                '4', '5', '6', '7', '8', '9', '10']),
        utils.to_dict(train_df,  'image.id', [
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']),
        BATCH_SIZE, model.preprocessing_function(), N_CLASSES, TARGET_SIZE)

    # Train model
    print("\ntraining model...")

    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None)
    model.change(base_model_trainable=False)

    print("\nTrain top model only...")
    history1 = pipeline.train_model_generator(model_target, model.model,
                                              train_generator, validation_generator, epochs=epochs_dense,
                                              early_stopping=None,
                                              force=FORCE_MODEL_FIT)

    print('==================')

    print("\nTrain top model all...")
    model.change(base_model_trainable=True, optimizer=Adam(
        lr=learning_rate_all,  decay=decay_all))
    history2 = pipeline.train_model_generator(model_target, model.model,
                                              train_generator, validation_generator, epochs=epochs_all,
                                              early_stopping=None, initial_epoch=epochs_dense,
                                              force=FORCE_MODEL_FIT, is_last=True)

    print('==================')
    training_time = time.time() - start
    pipeline.visualize_training(model_target, [history1, history2])

    # Test model
    print("\nmodel evaluation...")
    start = time.time()

    test_generator = pipeline.get_test_image_generator(IMAGE_PATH, utils.to_dict(
        test_df, 'image.id', ['1', '2', '3',
                              '4', '5', '6', '7', '8', '9', '10']),
        BATCH_SIZE, model.preprocessing_function(), N_CLASSES, TARGET_SIZE)

    pipeline.eval_generator(model_target, test_df, model.model, test_generator)

    eval_time = time.time() - start

    pipeline.save_timing(model_target, training_time, eval_time)
