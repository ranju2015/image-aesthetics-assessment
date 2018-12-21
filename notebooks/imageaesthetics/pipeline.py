import os.path
import os
import time
import pandas as pd
import numpy as np
import sklearn.model_selection as ms
from collections import Counter
from math import ceil
from . import dataset
from . import generators
from . import utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score

from keras.preprocessing.image import load_img, img_to_array


def load_data_set(source):
    ava = dataset.AVA(source)
    ava.download_metadata()
    ava.download_images()


def train_test_split(df, test_size=0.2, random_state=1):
    train_df, test_df = ms.train_test_split(
        df, test_size=test_size, random_state=random_state)
    return (train_df, test_df)


def load_df(target, path):
    train_df = pd.read_csv('{}/{}'.format(path, 'train.csv'), index_col=False)
    test_df = pd.read_csv('{}/{}'.format(path, 'test.csv'), index_col=False)

    print("loaded {} rows in training set".format(len(train_df)))
    print("loaded {} rows in test set".format(len(test_df)))

    test_df.to_csv("{}/test_data.csv".format(target), index=False)

    return (train_df, test_df)


def split_data(target, df, nb_samples_per_class=None, train_set_size=0.9, random_state=26, filter=None):
    train_df, validation_df = train_validation_under_sample_split(df,
                                                                  nb_samples_per_class=nb_samples_per_class,
                                                                  train_size=train_set_size, random_state=random_state)
    train_df.to_csv("{}/train_data.csv".format(target), index=False)
    validation_df.to_csv("{}/validation_data.csv".format(target), index=False)

    return (train_df, validation_df)


def train_validation_under_sample_split(df, nb_samples_per_class=None, train_size=0.8, random_state=1):
    train_df, test_df = ms.train_test_split(df, train_size=train_size,
                                            random_state=random_state)

    return (train_df, test_df)


def get_image_generators2(image_path, train_df, validation_df, batch_size, x_col, y_col,
                          target_size=None, hasExt=False, class_mode=None, rescale=1):

    train_generator = ImageDataGenerator(rescale=rescale).flow_from_dataframe(dataframe=train_df,
                                                                              directory=image_path, x_col=x_col, y_col=y_col,
                                                                              has_ext=hasExt, class_mode=class_mode, target_size=target_size,
                                                                              batch_size=batch_size, shuffle=False)

    validation_generator = ImageDataGenerator(rescale=rescale).flow_from_dataframe(dataframe=validation_df,
                                                                                   directory=image_path, x_col=x_col, y_col=y_col,
                                                                                   has_ext=hasExt, class_mode=class_mode, target_size=target_size,
                                                                                   batch_size=batch_size, shuffle=False)

    return (train_generator, validation_generator)


def get_test_image_generator2(image_path, test_df, batch_size, x_col, y_col,
                              target_size=None, hasExt=False, class_mode=None, rescale=1):
    test_generator = ImageDataGenerator(rescale=rescale).flow_from_dataframe(dataframe=test_df,
                                                                             directory=image_path, x_col=x_col, y_col=y_col,
                                                                             has_ext=hasExt, class_mode=class_mode,
                                                                             target_size=target_size,
                                                                             batch_size=batch_size, shuffle=False)
    return test_generator


def get_image_generators(image_path, train_samples, validation_samples, batch_size, basenet_preprocess, n_classes, target_size, img_format='jpg'):

    training_generator = generators.DataGenerator(
        train_samples, image_path, batch_size, n_classes, basenet_preprocess, img_format, img_crop_dims=target_size)

    validation_generator = generators.DataGenerator(
        validation_samples, image_path, batch_size, n_classes, basenet_preprocess, img_format,
        img_load_dims=target_size, img_crop_dims=None, shuffle=False)

    return (training_generator, validation_generator)


def get_test_image_generator(image_path, test_samples, batch_size, basenet_preprocess, n_classes, target_size, img_format='jpg'):

    return generators.DataGenerator(
        test_samples, image_path, batch_size, n_classes, basenet_preprocess, img_format,
        img_load_dims=target_size, img_crop_dims=None, shuffle=False)


def save_desc(target, desc):
    with open("{}/desc.txt".format(target), "w") as text_file:
        text_file.write(desc)

def save_timing(target, training_time, eval_time):
    with open("{}/timing.txt".format(target), "w") as text_file:
        text_file.write("Time for training: {}s, Model eval time: {}".format(training_time, eval_time))


def extract_train_validation_bottleneck_features(target, model,  train_generator,
                                                 validation_generator, force=False):
    train_features = get_bottleneck_features(
        target, model, train_generator, "train", force=force)

    validation_features = get_bottleneck_features(
        target, model, validation_generator, "validation", force=force)

    return(train_features, validation_features)


def extract_test_bottleneck_features(target, model, test_generator, force=False):
    test_features = get_bottleneck_features(
        target, model, test_generator, "test", force=force)

    return test_features


def get_bottleneck_features(target, model, generator, name, steps=None, force=False):
    filename = "{}/{}.npy".format(target, name)

    batch_size = generator.batch_size
    if steps == None:
        nb_samples = len(generator.filenames)
        steps = int(ceil(float(nb_samples)/float(batch_size)))

    if not force and os.path.isfile(filename):
        features = np.load(filename)
    else:
        print("{} steps for {} samples.".format(steps, nb_samples))
        start = time.time()

        features = model.predict_generator(generator, steps=steps)
        np.save(filename, features)

        print("{}s for predicting {} steps with batch size {}.".format(round(time.time() - start, 1),
                                                                       steps, batch_size))

    return features


def train_model(target, model,
                train_features,
                train_labels,
                validation_features,
                validation_labels,
                early_stopping=None,
                epochs=2500,
                shuffle=False,
                batch_size=30,
                force=False):
    model_arch_filename = "{}/model_arch.hdf5".format(target)
    model_weights_filename = "{}/model_weights.hdf5".format(target)

    if not os.path.exists(model_weights_filename) or force:
        model.save(model_arch_filename)
        checkpointer = ModelCheckpoint(filepath=model_weights_filename,
                                       verbose=1, save_best_only=True)
        callbacks = [checkpointer]
        if early_stopping != None:
            callbacks.append(early_stopping)

        history = model.fit(train_features, train_labels,
                            validation_data=(
                                validation_features, validation_labels),
                            epochs=epochs, callbacks=callbacks,
                            verbose=1, shuffle=shuffle, batch_size=batch_size)
        result = history.history
    else:
        train_features = None
        validation_features = None
        result = None

    return result


def train_model_generator(target, model,
                          train_generator,
                          validation_generator,
                          early_stopping=None,
                          epochs=2500,
                          shuffle=False,
                          batch_size=30, initial_epoch=0,
                          force=False, is_last=False):
    model_arch_filename = "{}/model_arch.hdf5".format(target)
    model_weights_filename = "{}/model_weights.hdf5".format(target)
    finished_filename = "{}/finished".format(target)

    if not os.path.exists(finished_filename) or force:
        model.save(model_arch_filename)
        checkpointer = ModelCheckpoint(filepath=model_weights_filename,
                                       verbose=1, save_best_only=True)
        callbacks = [checkpointer]
        if early_stopping != None:
            callbacks.append(early_stopping)

        history = model.fit_generator(train_generator,
                                      validation_data=validation_generator,
                                      epochs=epochs, callbacks=callbacks,
                                      verbose=1, shuffle=shuffle, 
                                      initial_epoch=initial_epoch, workers=1, use_multiprocessing=False)
        result = history.history
    else:
        result = None

    if is_last:
        with open(finished_filename, "w") as text_file:
            text_file.write("1")

    return result


def _merge(folder, histories):

    if None in histories:
        merged = pickle.load(open('{}/history.sav'.format(folder), "rb"))
    else:
        merged = {}
        for history in histories:
            for key, value in history.items():
                values = merged.get(key)
                if values != None:
                    values += value
                else:
                    values = value
                    merged[key] = values
        with open('{}/history.sav'.format(folder), 'wb') as file_pi:
            pickle.dump(merged, file_pi)

    return merged


def visualize_training(folder, histories, metric2=None):

    history = _merge(folder, histories)

    plt.figure(1, figsize=(20, 10))
    plt.subplot(121)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    if metric2 != None:
        plt.subplot(122)
        plt.plot(history[metric2])
        plt.plot(history['val_{}'.format(metric2)])
        plt.title('model {}'.format(metric2))
        plt.ylabel(metric2)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

    plt.savefig('{}/training_history.png'.format(folder))


def eval(target, model, test_features, test_labels):
    model_weights_filename = "{}/model_weights.hdf5".format(target)
    model.load_weights(model_weights_filename)
    score = model.evaluate(test_features, test_labels, verbose=0)

    print('Test score: {}'.format(score))
    return score


def eval_generator(target, test_df, model, generator):
    score_files =  "{}/test_scores.csv".format(target)
    if not os.path.exists(score_files):
        model_weights_filename = "{}/model_weights.hdf5".format(target)
        model.load_weights(model_weights_filename)
        score = model.evaluate_generator(generator, verbose=1)

        predictions = model.predict_generator(generator, verbose=1)
        values = utils.normalize_labels(
            list(test_df[['1','2','3','4','5','6','7','8','9', '10']].apply(lambda x: x.tolist(), axis=1)))
        pd.DataFrame(predictions, 
        columns=['1_pred','2_pred','3_pred','4_pred','5_pred','6_pred','7_pred','8_pred','9_pred', '10_pred']).to_csv('{}/test_prediction_data.csv'.format(target))


        y_mean = utils.means(values)
        y_pred_mean = utils.means(predictions)

        y_class = utils.bin_value(y_mean)
        y_pred_class = utils. bin_value(y_pred_mean)

        mse = mean_squared_error(y_mean, y_pred_mean)
        acc = accuracy_score(y_class, y_pred_class)

        pd.DataFrame(data={'acc': [acc], 'mse': [mse], 'emd': [score]}).to_csv(score_files)

        print('EMD: {}, ACC: {}, MSE: {}'.format(score, acc, mse))

        return score
    else:
        return None

    


def predict_generator(target, model, generator):
    model_weights_filename = "{}/model_weights.hdf5".format(target)
    model.load_weights(model_weights_filename)
    prediction = model.predict_generator(generator, verbose=0)

    return prediction


def predict(target, model, image, preprocess_input):

    model_weights_filename = "{}/model_weights.hdf5".format(target)
    model.load_weights(model_weights_filename)

    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return model.predict(x, batch_size=1, verbose=0)[0]


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: round(float(majority)/float(count), 2) for cls, count in counter.items()}
