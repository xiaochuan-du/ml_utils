#%%
import glob
import re

import numpy as np
import pandas as pd
from isoweek import Week
from pandas_summary import DataFrameSummary
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler


from keras.layers import Concatenate, Dense, Dropout, Embedding, Flatten, Input
from keras import initializers
from keras.models import Model

import utils
from importlib import reload
reload(utils)
# %load_ext autoreload
# %autoreload 2

#%%
ENV = 'dev'
DATA_DIR = r'/Users/kevindu/Documents/workspace/ml_utils/proj/RRVF/data'
if ENV == 'dev':
    # DEBUG
    import matplotlib.pyplot as plt
    from IPython.display import HTML
    # matplotlib inline

data = {
    'tra': pd.read_csv('{}/air_visit_data.csv'.format(DATA_DIR)),
    'as': pd.read_csv('{}/air_store_info.csv'.format(DATA_DIR)),
    'hs': pd.read_csv('{}/hpg_store_info.csv'.format(DATA_DIR)),
    'ar': pd.read_csv('{}/air_reserve.csv'.format(DATA_DIR)),
    'hr': pd.read_csv('{}/hpg_reserve.csv'.format(DATA_DIR)),
    'id': pd.read_csv('{}/store_id_relation.csv'.format(DATA_DIR)),
    'tes': pd.read_csv('{}/sample_submission.csv'.format(DATA_DIR)),
    'hol': pd.read_csv('{}/date_info.csv'.format(DATA_DIR))
}

# generate reserve info tbl
def get_reserve_tbl(data):
    " get_reserve_tbl "
    hpg_reserve = data['hr']
    hpg_store_info = data['hs']
    store_id_relation = data['id']
    air_reserve = data['ar']
    air_store_info = data['as']
    date_info = data['hol']

    hpg_joined = pd.merge(hpg_reserve, hpg_store_info, how='left', )
    hpg_fl_joined = pd.merge(store_id_relation, hpg_joined, how='left', )
    hpg_fl_joined.rename(
    {
        'hpb_latitude': 'latitude', 
        'hpb_longitude': 'longitude',
        'hpg_genre_name': 'genre_name',
        'hpg_area_name': 'area_name',
    }, axis='columns', inplace=True)
    hpg_fl_joined.drop('hpg_store_id', axis=1, inplace=True, errors="ignore")
    hpg_fl_joined = hpg_fl_joined.assign(src='hpg')

    air_joined = pd.merge(air_reserve, air_store_info, how='left', )
    air_fl_joined = air_joined
    air_fl_joined.rename(
        {
            'air_genre_name': 'genre_name',
            'air_area_name': 'area_name',
    }, axis='columns', inplace=True)
    air_fl_joined = air_fl_joined.assign(src='air')

    # clean reserve
    reserve = pd.concat([air_fl_joined, hpg_fl_joined], axis=0)
    reserve.visit_datetime = pd.to_datetime(reserve.visit_datetime )
    reserve.reserve_datetime = pd.to_datetime(reserve.reserve_datetime )

    date_info.drop('day_of_week', axis=1, inplace=True, errors="ignore")
    date_info.calendar_date = date_info.calendar_date.astype('str')
    reserve_en = reserve.assign(visit_date=reserve.visit_datetime.dt.date)
    reserve_en.visit_date = reserve_en.visit_date.astype('str')
    reserve_en = pd.merge(reserve_en, date_info, how='left',
        left_on=['visit_date'], right_on=['calendar_date'])
    reserve_en.rename(
        {
            'holiday_flg': 'visit_holiday_flg', 
    }, axis='columns', inplace=True)
    reserve_en.drop('calendar_date', axis=1, inplace=True, errors="ignore")

    reserve_en = reserve_en.assign(reserve_date=reserve_en.reserve_datetime.dt.date)
    reserve_en.reserve_date = reserve_en.reserve_date.astype('str')
    reserve_en = pd.merge(reserve_en, date_info, how='left',
        left_on=['reserve_date'], right_on=['calendar_date'])
    reserve_en.rename(
        {
            'holiday_flg': 'reserve_holiday_flg', 
    }, axis='columns', inplace=True)
    reserve_en.drop('calendar_date', axis=1, inplace=True, errors="ignore")
    data['reserve'] = reserve_en
    return data

data = get_reserve_tbl(data)
for tbl, df in data.items():
    print(tbl)
    display(df.head())

#%%
def get_store_stat_tbl(data):
    "get_store_stat_tbl"
    reserve = data['reserve']
    get_info_from_date(reserve, ['reserve_datetime', 'visit_datetime'])
    display(reserve[reserve.air_store_id == 'air_877f79706adbfb06'])
get_store_stat_tbl(data)

#%%
def tes2trn(tes):
    tes = tes.assign(air_store_id=tes["id"].map(lambda x: '_'.join(x.split('_')[:-1])))
    tes = tes.assign(visit_date=tes["id"].map(lambda x: x.split('_')[2]))
    return tes[["air_store_id", "visit_date"]]
def trn2test(tes_in_trn):
    tes_in_trn['id'] = df[['air_store_id', 'visit_date']].apply(lambda x: '_'.join(x), axis=1)
    return tes_in_trn[["id"]]
def get_info_from_date(data, dt_vars):
    "get_info_from_date"
    for dt_var in dt_vars:
        data[dt_var] = pd.to_datetime(data[dt_var])
        data["{}_week".format(dt_var)] = data[dt_var].dt.week
        data["{}_dayofweek".format(dt_var)] = data[dt_var].dt.dayofweek
        data["{}_year".format(dt_var)] = data[dt_var].dt.year
        data["{}_month".format(dt_var)] = data[dt_var].dt.month
trn = data['tra']
get_info_from_date(trn, ['visit_date'])

display(trn.head())

#%%
hol = data["hol"]
hol.rename(
    {
        'calendar_date': 'Date', 
}, axis='columns', inplace=True)
hol.Date = pd.to_datetime(hol.Date)
fld = 'holiday_flg'
hol = utils.add_ts_elapsed(fld, ['af_', 'be_'], hol)
hol = utils.add_ts_elapsed(fld, ['dur_'], hol)
display(hol)

#%%
# hol.Beforeholiday_flg.dtype 
# type(hol.holiday_flg)
store_info = data["reserve"][['air_store_id', "src",
    'genre_name', 'area_name', 'latitude', 'longitude']]
store_info.drop_duplicates(inplace=True)
air_store_info = store_info[store_info.src == 'air']
hpg_store_info = store_info[(store_info.src == 'hpg') & (~ store_info.genre_name.isna())]

hpg_store_info.rename(
    {
        'latitude': 'hpb_latitude', 
        'longitude': 'hpb_longitude',
        'genre_name': 'hpb_genre_name',
        'area_name': 'hpb_area_name',
    }, axis='columns', inplace=True)
hpg_store_info.drop('src', axis=1, inplace=True, errors="ignore")
# print(len(store_info))
# print(len(store_info.air_store_id.unique()))
store_info = pd.merge(air_store_info, hpg_store_info, how='left')
store_info.drop('src', axis=1, inplace=True, errors="ignore")

cate_vars = ['genre_name', 'area_name', 'hpb_area_name', 'hpb_genre_name', ]
conti_vars = ['latitude', 'longitude', 'hpb_latitude', 'hpb_longitude']
store_info[cate_vars] = store_info[cate_vars].fillna('UD')
store_info[conti_vars] = store_info[conti_vars].fillna(0)
display(store_info.head(20))

#%%
# import seaborn as sns
def trn2mat(trn_df, store_info, hol, cate_vars, conti_vars):
    trn = pd.merge(trn_df, store_info, how='left', on='air_store_id')
    str_date_hol = hol
    str_date_hol.Date = str_date_hol.Date.astype('str')
    trn = pd.merge(trn, hol, how='left', left_on='visit_date', right_on='Date')
    trn[cate_vars] = trn[cate_vars].fillna('UD')
    trn[conti_vars] = trn[conti_vars].fillna(0)
    get_info_from_date(trn, ['visit_date'])
    return trn.drop(['visit_date', 'Date', 'air_store_id'], axis=1)
trn = data['tra']
mat = trn2mat(trn, store_info, hol, cate_vars, conti_vars)
display(DataFrameSummary(mat).summary())

#%%
def mat2fea(mat):
    cat_vars = ['genre_name', 'area_name', 'hpb_genre_name', 
    'hpb_area_name', 'holiday_flg', 'dur_time_holiday_flg',
    'visit_date_week', 'visit_date_dayofweek', 'visit_date_year', 
    'visit_date_month']
    contin_vars = ['latitude', 'longitude', 'hpb_latitude', 'hpb_longitude',
            'af_holiday_flg', 'be_holiday_flg', 'dur_holiday_flg', 'dur_prog_holiday_flg']
    for v in contin_vars: mat.loc[mat[v].isnull(), v] = 0
    for v in cat_vars: mat.loc[mat[v].isnull(), v] = ""
    cat_maps = [(o, LabelEncoder()) for o in cat_vars]
    contin_maps = [([o], StandardScaler()) for o in contin_vars]


    cat_mapper = DataFrameMapper(cat_maps)
    cat_map_fit = cat_mapper.fit(mat)
    cat_cols = len(cat_map_fit.features)

    contin_mapper = DataFrameMapper(contin_maps)
    contin_map_fit = contin_mapper.fit(mat)
    contin_cols = len(contin_map_fit.features)
    def cat_preproc(dat):
        return cat_map_fit.transform(dat).astype(np.int64)

    def contin_preproc(dat):
        return contin_map_fit.transform(dat).astype(np.float)

    cat_map = cat_preproc(mat)
    contin_map = contin_preproc(mat)
    return cat_map, contin_map, cat_cols, contin_cols, cat_map_fit, mat.visitors

cat_map, contin_map, cat_cols, contin_cols, cat_map_fit, y = mat2fea(mat)
print('Done')
#%%

def ts_data_split(datas):
    output = {
        'trn': [],
        'valid': []
    }
    train_ratio = 0.9
    size = datas[0].shape[0]
    trn_size = int(train_ratio * size)
    for data in datas:
        output['trn'].append(data[:trn_size])
        output['valid'].append(data[trn_size:])
    return output

ts_data = ts_data_split([cat_map, contin_map, y])
cat_map_train, contin_map_train, y_train_orig = ts_data['trn']
cat_map_valid, contin_map_valid, y_valid_orig = ts_data['valid']

print(y_valid_orig.shape)

#%%
y_train, y_valid, max_log_y = uniform_y(y_train_orig, y_valid_orig)
def uniform_y(y_train_orig, y_valid_orig):
    max_log_y = max(np.max(np.log(y_train_orig)), np.max(np.log(y_valid_orig)))
    return np.log(y_train_orig)/max_log_y, np.log(y_valid_orig)/max_log_y, max_log_y

def rmsle(y_pred, targ = y_valid_orig):
    log_vars = np.log(targ + 1) - np.log(y_pred + 1)
    return math.sqrt(np.square(log_vars).mean())

def log_max_inv(preds, mx=max_log_y):
    return np.exp(preds * mx)

def my_init(scale):
    return lambda shape, name=None: initializations.uniform()


def emb_init(shape, name=None):
    return initializers.RandomUniform()


def cat_map_info(feat):
    return feat[0], len(feat[1].classes_)

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
    x = Dense(1, activation='sigmoid')(x)

    model = Model([inp for inp, emb in embs] + [contin_inp], x)
    model.compile('adam', 'mse')
    return model

def split_cols(arr): return np.hsplit(arr,arr.shape[1])
map_train = split_cols(cat_map_train) + [contin_map_train]
map_valid = split_cols(cat_map_valid) + [contin_map_valid]

model = get_model(contin_cols, cat_map_fit)
print('Done')
# map_train = split_cols(cat_map_train) + split_cols(contin_map_train)
# map_valid = split_cols(cat_map_valid) + split_cols(contin_map_valid)

#%%
model.optimizer.lr = 1e-4
model.fit(
    map_train,
    y_train,
    batch_size=128,
    epochs=2,
    validation_data=(map_valid, y_valid))

#%%
print('hi')
# model.evaluate(map_valid, y_valid)

#%%
y_pred = model.predict(map_valid)
# rmsle(y_pred, targ = y_valid_orig)
a = log_max_inv(y_pred, mx=max_log_y)
diff = a - y_valid_orig
# print(pred)

#%%
plt.plot(y_valid_orig)

#%%
display(DataFrameSummary(trn).summary())