import os
import numpy as np
import keras
from . import utils


class DataGenerator(keras.utils.Sequence):
    def __init__(self, samples, img_dir, batch_size, n_classes, basenet_preprocess, img_format,
                 img_load_dims=(256, 256), img_crop_dims=(224, 224), shuffle=True, label_col='labels', id_col='image.id'):
        self.samples = samples
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.basenet_preprocess = basenet_preprocess
        self.img_load_dims = img_load_dims
        self.img_crop_dims = img_crop_dims
        self.shuffle = shuffle
        self.img_format = img_format
        self.on_epoch_end()
        self.label_col = label_col
        self.id_col = id_col

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index *
                                     self.batch_size:(index+1)*self.batch_size]
        batch_samples = [self.samples[i] for i in batch_indexes]
        X, y = self.__data_generator(batch_samples)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generator(self, batch_samples):
        dim = self.img_load_dims
        if self.img_crop_dims != None:
            dim = self.img_crop_dims

        X = np.empty((len(batch_samples), * dim, 3))
        y = np.empty((len(batch_samples), self.n_classes))

        for i, sample in enumerate(batch_samples):
            img_file = os.path.join(self.img_dir, '{}.{}'.format(
                sample[self.id_col], self.img_format))
            img = utils.load_image(img_file, self.img_load_dims)
            if img is not None:

                if self.img_crop_dims != None:
                    img = utils.random_crop(img, dim)
                    img = utils.random_horizontal_flip(img)
                X[i, ] = img

            y[i, ] = utils.normalize_labels(sample[self.label_col])

        X = self.basenet_preprocess(X)

        return X, y
