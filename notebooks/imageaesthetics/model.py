
import importlib

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, InputLayer


class AestheticsModel:

    def __init__(self, base_model_name, top_model, input_shape,
                 optimizer, loss, base_model_weights='imagenet',):

        index = base_model_name.rfind('.')
        model_name = base_model_name[index+1:]
        model_package = base_model_name[:index]
        self.basel_model_package = importlib.import_module(
            model_package)

        self.base_model_name = model_name
        self.base_model_weights = base_model_weights
        self.input_shape = input_shape
        self.top_model = top_model
        self.top_model_n_layers = 0
        for layer in top_model.layers:
            self.top_model_n_layers += 1

        self.optimizer = optimizer
        self.loss = loss

        self._build()
        self._compile()

    def _build(self):
        BaseModel = getattr(self.basel_model_package, self.base_model_name)

        self.base_model = BaseModel(input_shape=self.input_shape,
                                    weights=self.base_model_weights, include_top=False, pooling='avg')

        self.base_model.layers.insert(
            0, InputLayer(input_shape=self.input_shape))
        for layer in self.top_model.layers:
            self.base_model.layers.append(layer)
        self.model = Sequential(layers=self.base_model.layers)

    def _compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def preprocessing_function(self):
        return getattr(self.basel_model_package, 'preprocess_input')

    def _basemodel_layers_trainable(self, trainable):
        for layer in self.model.layers[:-self.top_model_n_layers]:
            layer.trainable = trainable

    def _change_optimizer(self, optimizer):
        self.optimizer = optimizer

    def change(self, base_model_trainable=False, optimizer=None):
        self._basemodel_layers_trainable(base_model_trainable)
        if optimizer != None:
            self._change_optimizer(optimizer)

        self._compile()
