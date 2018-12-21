
import os
import json
import keras
import numpy as np
import pandas as pd


def to_dict(df, id, value_vars):
    result = []
    for index, row in df.iterrows():
        labels = []
        for value_var in value_vars:
            labels.append(row[value_var])
        result.append({id: row[id], 'labels': labels})

    return result


def random_crop(img, crop_dims):
    h, w = img.shape[0], img.shape[1]
    ch, cw = crop_dims[0], crop_dims[1]
    assert h >= ch
    assert w >= cw
    x = np.random.randint(0, w - cw + 1)
    y = np.random.randint(0, h - ch + 1)
    return img[y: (y+ch), x: (x+cw), :]


def random_horizontal_flip(img):
    assert len(img.shape) == 3
    assert img.shape[2] == 3
    if np.random.random() < 0.5:
        img = img.swapaxes(1, 0)
        img = img[:: -1, ...]
        img = img.swapaxes(0, 1)
    return img


def load_image(img_file, target_size):
    return np.asarray(keras.preprocessing.image.load_img(img_file, target_size=target_size))


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist*np.arange(1, 11)).sum()


def means(distributions):
    mean_scores = []
    for distribution in distributions:
        mean_scores.append(calc_mean_score(distribution))
    return mean_scores


def bin_value(values):
    bins = []
    for value in values:
        if value <= 5:
            bins.append("low")
        else:
            bins.append("high")
    return bins


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
