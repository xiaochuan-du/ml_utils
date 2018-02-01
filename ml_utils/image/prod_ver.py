import pandas as pd
from pandas_summary import DataFrameSummary
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
import numpy as np
import pickle
from sklearn_pandas import DataFrameMapper
from keras.layers import Concatenate, Dense, Dropout, Embedding, Flatten, Input
from keras import initializers
from keras.models import Model


def split_cols(arr):
    return np.hsplit(arr, arr.shape[1])


def cat_map_info(feat):
    return feat[0], len(feat[1].classes_)


def cat_preproc(dat):
    return cat_map_fit.transform(dat).astype(np.int64)


def contin_preproc(dat):
    return contin_map_fit.transform(dat).astype(np.float32)


def my_init(scale):
    return lambda shape, name=None: initializations.uniform()


def emb_init(shape, name=None):
    return initializers.RandomUniform()


def get_emb(feat):
    name, c = cat_map_info(feat)
    c2 = (c + 1) // 2
    if c2 > 50: c2 = 50
    inp = Input((1, ), dtype='int64', name=name + '_in')
    # , W_regularizer=l2(1e-6)
    u = Flatten(name=name + '_flt')(Embedding(
        c, c2, input_length=1)(inp))  # , init=emb_init
    #     u = Flatten(name=name+'_flt')(Embedding(c, c2, input_length=1)(inp))
    return inp, u


def get_contin(feat):
    name = feat[0][0]
    inp = Input((1, ), name=name + '_in')
    return inp, Dense(1, name=name + '_d')(inp)  # , init=my_init(1.)


def get_model(contin_cols, cat_map_fit):
    contin_inp = Input((contin_cols, ), name='contin')
    contin_out = Dense(
        contin_cols * 10, activation='relu', name='contin_d')(contin_inp)
    #contin_out = BatchNormalization()(contin_out)
    embs = [get_emb(feat) for feat in cat_map_fit.features]
    #conts = [get_contin(feat) for feat in contin_map_fit.features]
    #contin_d = [d for inp,d in conts]
    x = Concatenate()([emb for inp, emb in embs] + [contin_out])

    x = Dropout(0.02)(x)
    x = Dense(1000, activation='relu', kernel_initializer='uniform')(x)
    x = Dense(500, activation='relu', kernel_initializer='uniform')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='relu')(x)

    model = Model([inp for inp, emb in embs] + [contin_inp], x)
    #model = Model([inp for inp,emb in embs] + [inp for inp,d in conts], x)
    model.compile('adam', 'mse')
    #model.compile(Adam(), 'mse') mean_absolute_error
    return model


def get_mul_out_model(contin_cols, y_vars, cat_map_fit):
    contin_inp = Input((contin_cols, ), name='contin')
    contin_out = Dense(
        contin_cols * 10, activation='relu', name='contin_d')(contin_inp)
    #contin_out = BatchNormalization()(contin_out)
    embs = [get_emb(feat) for feat in cat_map_fit.features]
    #conts = [get_contin(feat) for feat in contin_map_fit.features]
    #contin_d = [d for inp,d in conts]
    x = Concatenate()([emb for inp, emb in embs] + [contin_out])

    x = Dropout(0.02)(x)
    x = Dense(1000, activation='relu', kernel_initializer='uniform')(x)
    x = Dense(500, activation='relu', kernel_initializer='uniform')(x)
    x = Dropout(0.2)(x)
    outs = [Dense(1, activation='relu', name=col)(x) for col in y_vars]
    model = Model([inp for inp, emb in embs] + [contin_inp], outs)
    #model = Model([inp for inp,emb in embs] + [inp for inp,d in conts], x)
    model.compile('adam', 'mse')
    #model.compile(Adam(), 'mse') mean_absolute_error
    return model


cat_cols = 9
contin_cols = 65
test = pd.read_pickle('./major40_af_etl.pkl')
test.fillna(0, inplace=True)
y_vars = [col for col in test.columns if col.startswith('next')]
x_vars = [col for col in test.columns if col not in y_vars]
test_x, test_y = test[x_vars], test[y_vars]
val = test_x['2017-04':]
trn = test_x['2017-01-10':'2017-08']
y_trn_tt = [test_y[col]['2017-01-10':'2017-08'] for col in y_vars]
y_valid_tt = [test_y[col]['2017-04':] for col in y_vars]
# y_vars = ['next_1_A_sum', 'next_1_AB_sum']
cat_map_fit = pickle.load(open('cat_maps.pickle', 'rb'))
contin_map_fit = pickle.load(open('contin_maps.pickle', 'rb'))

y_vars = ['next_7_TOTAL_sum']
model = get_mul_out_model(contin_cols, y_vars, cat_map_fit)


#y_trn = test_y.next_1_A_sum['2011-01-10':'2017-03']
#y_valid = test_y.next_1_A_sum['2017-03':]
def preprocessing(df):
    cat_map = cat_preproc(df)
    contin = contin_preproc(df)
    df_map = split_cols(cat_map) + [contin]
    return df_map


map_train = preprocessing(trn)
map_valid = preprocessing(val)
model.optimizer.lr = 1e-3
model.fit(
    map_train,
    y_trn_tt[0],
    batch_size=128,
    epochs=50,
    validation_data=(map_valid, map_valid[0]))
model.save_weights('caching.h5')
#model.fit(map_train, [y_trn_tt[0], y_trn_tt[1], y_trn_tt[2], y_trn_tt[3], y_trn_tt[4], y_trn_tt[5], y_trn_tt[6], y_trn_tt[7], y_trn_tt[8], y_trn_tt[9], y_trn_tt[10], y_trn_tt[11], y_trn_tt[12], y_trn_tt[13], y_trn_tt[14], y_trn_tt[15], y_trn_tt[16], y_trn_tt[17], y_trn_tt[18], y_trn_tt[19]], batch_size=128, epochs=20, verbose=1, validation_data=(map_valid, [y_valid_tt[0], y_valid_tt[1], y_valid_tt[2], y_valid_tt[3], y_valid_tt[4], y_valid_tt[5], y_valid_tt[6], y_valid_tt[7], y_valid_tt[8], y_valid_tt[9], y_valid_tt[10], y_valid_tt[11], y_valid_tt[12], y_valid_tt[13], y_valid_tt[14], y_valid_tt[15], y_valid_tt[16], y_valid_tt[17], y_valid_tt[18], y_valid_tt[19]]))
