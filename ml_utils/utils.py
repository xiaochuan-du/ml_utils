"""
    utils functions
"""
import math

import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import deserialize as layer_from_config
from keras.layers import serialize
from keras.preprocessing import image, sequence
from keras.utils.np_utils import to_categorical

import bcolz
import PIL
from PIL import Image


def gray(img):
    " from colorful images to gray ones"
    to_bw = np.array([0.299, 0.587, 0.114])
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).dot(to_bw)
    else:
        return np.rollaxis(img, 0, 3).dot(to_bw)

def floor(val):
    " floor "
    return int(math.floor(val))

def ceil(val):
    " ceil "
    return int(math.ceil(val))


def do_clip(arr, max_val):
    " tbd "
    clipped = np.clip(arr, (1-max_val)/1, max_val)
    return clipped/clipped.sum(axis=1)[:, np.newaxis]


def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(224,224)):
    " get image batches from directory, default action is to resize images directly "
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


def onehot(cls_vec):
    " onehot y or features"
    return to_categorical(cls_vec)


def adjust_dropout(weights, prev_p, new_p):
    " tbd "
    scal = (1-prev_p)/(1-new_p)
    return [o*scal for o in weights]


def get_data(path, target_size=(224,224)):
    """
        get data as matrics, eg:
        trn = get_data(path+'train')
    """
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.samples)])


def save_array(fname, arr):
    " save np matrix or array"
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    " load np matrix or array"
    return bcolz.open(fname)[:]


def mk_size(img, r2c):
    " padding image to get new row to col ratio"
    r,c,_ = img.shape
    curr_r2c = r/c
    new_r, new_c = r,c
    if r2c>curr_r2c:
        new_r = floor(c*r2c)
    else:
        new_c = floor(r/r2c)
    arr = np.zeros((new_r, new_c, 3), dtype=np.float)
    r2=(new_r-r)//2
    c2=(new_c-c)//2
    arr[floor(r2):floor(r2)+r,floor(c2):floor(c2)+c] = img
    return arr


def mk_square(img):
    " padding image to get a square one"
    x,y,_ = img.shape
    maxs = max(img.shape[:2])
    y2=(maxs-y)//2
    x2=(maxs-x)//2
    arr = np.zeros((maxs,maxs,3), dtype=np.float)
    arr[floor(x2):floor(x2)+x,floor(y2):floor(y2)+y] = img
    return arr


def get_classes(path):
    """
    be noticed that test batches, more often than not do not have labels,
    and get_data is to get matrices, while this api is to get generator
    (val_classes, trn_classes, val_labels, trn_labels,
    val_filenames, filenames, test_filenames) = get_classes(path)
    """
    trn_batches = get_batches(path+'train', shuffle=False, batch_size=1)
    val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
    test_batches = get_batches(path+'test', shuffle=False, batch_size=1)
    return (val_batches.classes, trn_batches.classes, onehot(val_batches.classes), onehot(trn_batches.classes),
        val_batches.filenames, trn_batches.filenames, test_batches.filenames)





class MixIterator(object):
    """
    Mix iter through a couple of generators, eg:
        mi = MixIterator([batches, test_batches, val_batches])
        bn_model.fit_generator(mi, mi.N, nb_epoch=8, validation_data=(conv_val_feat, val_labels))
    """
    def __init__(self, iters):
        self.iters = iters
        self.multi = type(iters) is list
        if self.multi:
            self.N = sum([it[0].N for it in self.iters])
        else:
            self.N = sum([it.N for it in self.iters])

    def reset(self):
        for lc_iter in self.iters:
            lc_iter.reset()

    def __iter__(self):
        return self

    def next(self, *args, **kwargs):
        if self.multi:
            nexts = [[next(it) for it in o] for o in self.iters]
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
            return (n0, n1)
        else:
            nexts = [next(it) for it in self.iters]
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
            return (n0, n1)
