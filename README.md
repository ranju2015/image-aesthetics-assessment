# Machine Learning Nano Degree Capstone Project: "Image Aesthetics Assessment"

## Dataset 

The project uses the AVA dataset which can be downloaded here (32 GB):

[AVA dataset](http://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460)

The metadata is hosted on Github and will be checked out at the beginning of the training.

## Report for the Project

[Project Report](report.pdf)

## Notebooks/Scripts

The scripts are located in the notebook folder. 
It holds all notebooks for the different datasets which were evaluated + the training script:

[notebooks](notebooks)

The overall training shell script with all the model parameters

[notebooks/trainings.sh](notebooks/trainings.sh)

The main training script:

[notebooks/trainings.py](notebooks/train.py)

Code modules:

[notebooks/imageaesthetics](notebooks/imageaesthetics)


## Best model

The best model (architecture and weights file) can downloaded from S3:

[Model architecture](https://s3.amazonaws.com/aesthetics-88h7ezehezz2/model_arch.hdf5)

[Model weights](https://s3.amazonaws.com/aesthetics-88h7ezehezz2/model_weights.hdf5)

For using the model Keras 2.2.4 needs to be installed.
The model can be used like this:

```python


def earth_mover_loss(y_true, y_pred):
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
    return K.mean(emd)


import keras.losses
keras.losses.earth_mover_loss = earth_mover_loss

model = load_model("{}/model_arch.hdf5".format(dir))
model_weights_filename = "{}/model_weights.hdf5".format(dir)
model.load_weights(model_weights_filename)
```
