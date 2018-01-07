"""
    code about transfer learning and fine tuning
"""

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from ml_utils.utils import layer_from_config, serialize


def get_dense_layers(input_layers):
    " get dense layers on top"
    return [
        MaxPooling2D(input_shape=input_layers.output_shape[1:]),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(1000, activation='softmax')
        ]

def wrap_config(layer):
    " get dict of layer's name and its config"
    return serialize(layer)


def copy_layer(layer):
    " copy layer w/p weighting"
    return layer_from_config(wrap_config(layer))


def copy_layers(layers):
    " copy layers w/o weighting"
    """
     example
     new_layers = copy_layers(bn_model.layers)
     for layer in new_layers:
     conv_model.add(layer)
     copy_weights(bn_model.layers, new_layers)
     conv_model.compile(Adam(1e-5), 'categorical_crossentropy', ['accuracy'])
    """
    return [copy_layer(layer) for layer in layers]


def copy_weights(from_layers, to_layers):
    "copy weights"
    for from_layer, to_layer in zip(from_layers, to_layers):
        to_layer.set_weights(from_layer.get_weights())


def copy_model(model):
    "copy model and weights"
    res = Sequential(copy_layers(model.layers))
    copy_weights(model.layers, res.layers)
    return res


def insert_layer(model, new_layer, index):
    " insert layer as position index, other layer keep config and wghts"
    res = Sequential()
    for i, layer in enumerate(model.layers):
        if i == index:
            res.add(new_layer)
        copied = layer_from_config(wrap_config(layer))
        res.add(copied)
        copied.set_weights(layer.get_weights())
    return res

def split_at(model, layer_type):
    layers = model.layers
    layer_idx = [index for index,layer in enumerate(layers)
                 if type(layer) is layer_type][-1]
    return layers[:layer_idx+1], layers[layer_idx+1:]

def vgg_ft(out_dim):
    vgg = Vgg16()
    vgg.ft(out_dim)
    model = vgg.model
    return model

def vgg_ft_bn(out_dim):
    vgg = Vgg16BN()
    vgg.ft(out_dim)
    model = vgg.model
    return model
