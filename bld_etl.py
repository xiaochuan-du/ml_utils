import datetime
import math
import operator
import pickle
import re
import random
import bcolz
import keras
import keras.backend as K
import matplotlib.pyplot as plt  # xgboost
import numpy as np
import pandas as pd
from IPython.display import HTML, Audio, display
from isoweek import Week
from keras import Model, initializers
from keras.layers import Concatenate, Dense, Dropout, Embedding, Flatten, Input
from pandas_summary import DataFrameSummary
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
import logging
logger = logging.getLogger(__name__)


def read_raw(merged_df):
    y_cols = [col for col in merged_df.columns if col.startswith('AF_')]
    x_cols = [col for col in merged_df.columns if col.startswith('BE_')]
    y_cols.append('_id'), x_cols.append('_id')
    y_cols.extend(
        [col for col in merged_df.columns if col.startswith('TRANS_')])
    y_cols.append('RBC_LOSS')
    ignore_cols = ['dt_data', 'dt_now']
    x_cols.extend(
        list(
            set(merged_df.columns) - (set(x_cols) | set(y_cols)) -
            set(ignore_cols)))
    x_cols.append('TRANS_RBC')
    return merged_df[y_cols], merged_df[x_cols]


def loadtypes(data_df):
    summary_df = DataFrameSummary(data_df).summary()
    # auto evaluate datatype
    contin_vars = [
        col for col in summary_df.columns
        if summary_df.loc["types"][col] == 'numeric'
    ]
    bool_vars = [
        col for col in summary_df.columns
        if summary_df.loc["types"][col] == 'bool'
    ]
    cat_vars = [
        col for col in summary_df.columns
        if summary_df.loc["types"][col] == 'categorical'
    ]
    dt_vars = [
        col for col in summary_df.columns
        if summary_df.loc["types"][col] == 'date'
    ]
    const_vars = [
        col for col in summary_df.columns
        if summary_df.loc["types"][col] == 'constant'
    ]
    text_vars = []
    for var in ['DEPT_CODE']:
        contin_vars.remove(var), cat_vars.append(var)
    for var in ['OP_NAME', 'INHOS_DIAG_NAME']:
        cat_vars.remove(var), text_vars.append(var)
    cat_vars.extend(bool_vars)
    return contin_vars, cat_vars, dt_vars, text_vars, const_vars


def add_datepart(data_df, cols):
    " add_datepart "
    for col in cols:
        data_df[col] = pd.to_datetime(data_df[col], unit='ms')
        data_df["{}_Year".format(col)] = data_df[col].dt.year
        data_df["{}_Month".format(col)] = data_df[col].dt.month
        data_df["{}_Week".format(col)] = data_df[col].dt.week
        data_df["{}_Day".format(col)] = data_df[col].dt.day
        data_df["{}_Hour".format(col)] = data_df[col].dt.hour


def fillna(data_df, contin_vars, cat_vars):
    " Next we'll fill in missing values to avoid complications w/ na's. "
    data_df[contin_vars] = data_df[contin_vars].fillna(-999)
    data_df[cat_vars] = data_df[cat_vars].fillna('_NAN')


def type_map(data_df, cat_vars, contin_vars, cache_dir, use_cache=True):
    " type_map "
    if use_cache:
        cat_map_fit = pickle.load(
            open('{}/cat_maps.pickle'.format(cache_dir), 'rb'))
        contin_map_fit = pickle.load(
            open('{}/contin_maps.pickle'.format(cache_dir), 'rb'))
    else:
        cat_maps = [(o, LabelEncoder()) for o in cat_vars]
        contin_maps = [([o], StandardScaler()) for o in contin_vars]
        cat_mapper = DataFrameMapper(cat_maps)
        cat_map_fit = cat_mapper.fit(data_df)
        contin_mapper = DataFrameMapper(contin_maps)
        contin_map_fit = contin_mapper.fit(data_df)
        pickle.dump(contin_map_fit,
                    open('{}/contin_maps.pickle'.format(cache_dir), 'wb'))
        pickle.dump(cat_map_fit,
                    open('{}/cat_maps.pickle'.format(cache_dir), 'wb'))
    cat_cols = len(cat_map_fit.features)
    contin_cols = len(contin_map_fit.features)
    logger.info("cat_cols: {}, contin_cols: {}".format(cat_cols, contin_cols))
    logger.info("cat_map_fit.features: {}".format(
        [len(o[1].classes_) for o in cat_map_fit.features]))
    return cat_map_fit, contin_map_fit, contin_cols

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]

def cat_preproc(dat, cat_map_fit):
    "cat_preproc"
    return cat_map_fit.transform(dat).astype(np.int64)


def contin_preproc(dat, contin_map_fit):
    "contin_preproc"
    return contin_map_fit.transform(dat).astype(np.float32)


def get_data(joined, cat_map_fit, contin_map_fit):
    n = len(joined)
    samp_size = 5000
    np.random.seed(42)
    idxs = sorted(np.random.choice(n, samp_size, replace=False))

    train_ratio = 0.9
    train_size = int(samp_size * train_ratio)
    joined_samp = joined.iloc[idxs]  # .set_index("OP_DTIME")
    joined_valid = joined_samp[train_size:]
    joined_train = joined_samp[:train_size]
    y_train_orig = joined_train.TRANS_RBC
    y_valid_orig = joined_valid.TRANS_RBC
    joined_train.drop('TRANS_RBC', axis=1, inplace=True)
    joined_valid.drop('TRANS_RBC', axis=1, inplace=True)
    logger.info("data_size: {}, train_size: {}".format(n, train_size))
    cat_map_train = cat_preproc(joined_train, cat_map_fit)
    cat_map_valid = cat_preproc(joined_valid, cat_map_fit)
    contin_map_train = contin_preproc(joined_train, contin_map_fit)
    contin_map_valid = contin_preproc(joined_valid, contin_map_fit)

    x_train = np.concatenate((cat_map_train, contin_map_train), axis=1)
    x_valid = np.concatenate((cat_map_valid, contin_map_valid), axis=1)
    return x_train, y_train_orig, x_valid, y_valid_orig, cat_map_train, cat_map_valid, contin_map_train, contin_map_valid

def from_fasttest_to_vectors(file_name):
    with open(file_name, 'r') as f:
        data = f.readlines()
    samples, dim = data[0].split()
    samples, dim = int(samples), int(dim)
    wordidx = {}
    vecs = np.zeros((samples, dim), dtype='float32')
    for idx, line in enumerate(data[1:]):
        word, vec = line.split(' ', 1)
        vecs[idx] = np.array([ float(i) for i in vec.split()])
        wordidx[word] = idx
    return vecs, list(wordidx.keys()), wordidx

def dump_vectors(loc, vecs, words, wordidx):
    return (save_array(loc+'.dat', vecs),
        pickle.dump(words, open(loc+'_words.pkl','wb')),
        pickle.dump(wordidx, open(loc+'_idx.pkl','wb')) )


def load_vectors(loc):
    return (load_array(loc+'.dat'),
        pickle.load(open(loc+'_words.pkl','rb')),
        pickle.load(open(loc+'_idx.pkl','rb')))
# dump_vectors('/root/.keras/models/fsttxtzh300', vecs, words, wordidx)
# vecs, words, wordidx = from_fasttest_to_vectors('/root/.keras/models/fsttxtzh300')

def dump_var(loc, var):
    pickle.dump(var, open(loc+'.pkl','wb'))

def load_var(loc):
    return pickle.load(open(loc+'.pkl','rb'))

def create_emb(vocab_size, vecs, words):
    words_set = set(words_set)
    n_fact = vecs.shape[1]
    emb = np.zeros((vocab_size, n_fact))

    for i in range(1,len(emb)):
        word = idx2word[i]
        if re.match(r"^[A-Z]*$", word):
            word = word.lower()
        if word and (word in words_set):
            src_idx = wordidx[word]
            emb[i] = vecs[src_idx]
        else:
            # If we can't find the word in glove, randomly initialize
            emb[i] = normal(scale=0.6, size=(n_fact,))

    # This is our "rare word" id - we want to randomly initialize
    emb[-1] = normal(scale=0.6, size=(n_fact,))
    emb/=3
    return emb

def par2idx(para, words, wordidx):
    words_set = set(words)
    paras = []
    for word in para:
        if re.match(r"^[A-Z]*$", word):
            word = word.lower()
        if word and (word in words_set):
            paras.append(wordidx[word])
        else:
            paras.append(-1)
    return np.array(paras, dtype='int32')
