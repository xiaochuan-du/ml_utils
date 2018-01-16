import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler


from keras.layers import Concatenate, Dense, Dropout, Embedding, Flatten, Input, BatchNormalization
from keras import initializers
from keras.models import Model

cate_vars = ['genre_name', 'area_name', 'hpb_area_name', 'hpb_genre_name', ]
conti_vars = ['latitude', 'longitude', 'hpb_latitude', 'hpb_longitude']


class TimeToEvent(object):
    'iter across row'
    def __init__(self, fld, init_date):
        '__init__'
        self.fld = fld
        self.last = init_date
        
    def get(self, row):
        'getter'
        if (row[self.fld] == 1):
            self.last = row.Date
        return row.Date-self.last

class DurationTime(object):
    'iter across row'
    def __init__(self, fld):
        '__init__'
        self.fld = fld
        self.dura = 0
        
    def get(self, row):
        'getter'
        if (row[self.fld] == 1):
            self.dura = self.dura + 1
        else:
            self.dura = 0
        return self.dura

class Duration(object):
    'iter across row'
    def __init__(self, fld):
        self.fld = fld
        self.dura = 0
        self.past = 0
        
    def get(self, row):
        if (row[self.fld] != 0 and self.past != 0):
            self.dura = self.dura
        elif (row[self.fld] != 0 and self.past == 0):
            self.dura = row[self.fld]
        else:
            self.dura = 0
        self.past = row[self.fld]
        return self.dura

def add_ts_elapsed(fld, prefixs, df):
    "add_elapsed"
    if len(prefixs) == 2:
        # bi-conditions
        prefix = prefixs[0]
        df = df.sort_values(['Date'])
        init_date = df[df[fld] == 1].iloc[0]['Date']
        sh_el = TimeToEvent(fld, init_date)
        df[prefix+fld] = df.apply(sh_el.get, axis=1).dt.days

        prefix = prefixs[-1]
        df = df.sort_values(['Date'], ascending=[False])
        init_date = df[df[fld] == 1].iloc[0]['Date']
        sh_el = TimeToEvent(fld, init_date)
        df[prefix+fld] = df.apply(sh_el.get, axis=1).dt.days
        df = df.sort_values(['Date'])
        return df
    else:
        # duration
        prefix = prefixs[0]
        dt_fld = prefix + "time_" + fld
        dur_fld = prefix + fld
        prog_fld = prefix + "prog_" + fld

        df = df.sort_values(['Date'])
        sh_el = DurationTime(fld)
        df[dt_fld] = df.apply(sh_el.get, axis=1)
        prefix = prefixs[0]
        df = df.sort_values(['Date'], ascending=[False])
        sh_el = Duration(dt_fld)
        df[dur_fld] = df.apply(sh_el.get, axis=1)
        df = df.sort_values(['Date'])
        df[prog_fld] = df[dt_fld] / df[dur_fld]
        df[prog_fld].fillna(0, inplace=True)
        return df

def get_store_stat_tbl(data):
    "get_store_stat_tbl"
    reserve = data['reserve']
    get_info_from_date(reserve, ['reserve_datetime', 'visit_datetime'])

def get_info_from_date(data, dt_vars):
    "get_info_from_date"
    for dt_var in dt_vars:
        data[dt_var] = pd.to_datetime(data[dt_var])
        data["{}_week".format(dt_var)] = data[dt_var].dt.week
        data["{}_dayofweek".format(dt_var)] = data[dt_var].dt.dayofweek
        data["{}_year".format(dt_var)] = data[dt_var].dt.year
        data["{}_month".format(dt_var)] = data[dt_var].dt.month

def tes2trn(tes):
    tes = tes.assign(air_store_id=tes["id"].map(lambda x: '_'.join(x.split('_')[:-1])))
    tes = tes.assign(visit_date=tes["id"].map(lambda x: x.split('_')[2]))
    return tes[["air_store_id", "visit_date"]]

def trn2test(tes_in_trn):
    tes_in_trn['id'] = df[['air_store_id', 'visit_date']].apply(lambda x: '_'.join(x), axis=1)
    return tes_in_trn[["id"]]

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

def trn2mat(trn_df, store_info, hol, cate_vars, conti_vars):
    " from train like data to mat"
    trn = pd.merge(trn_df, store_info, how='left', on='air_store_id')
    str_date_hol = hol
    str_date_hol.Date = str_date_hol.Date.astype('str')
    trn = pd.merge(trn, str_date_hol, how='left', left_on='visit_date', right_on='Date')
    trn[cate_vars] = trn[cate_vars].fillna('UD')
    trn[conti_vars] = trn[conti_vars].fillna(0)
    get_info_from_date(trn, ['visit_date'])
    return trn.drop(['visit_date', 'Date', 'air_store_id'], axis=1)

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

def ts_data_split(input_map, y, s_i, e_i):
    output = {
        'trn': [],
        'valid': []
    }
    # train_ratio = 0.9
    # size = y.shape[0]
    # trn_size = int(train_ratio * size)
    input_trn = []
    input_valid = []
    y_trn = np.concatenate((y.iloc[:s_i].values, y.iloc[e_i:].values), axis=0)
    y_valid = y.iloc[s_i: e_i]
    for fea in input_map:
        input_valid.append(fea[s_i:e_i])
        input_trn.append( np.concatenate((fea[:s_i], fea[e_i:]), axis=0) )

    return input_trn, input_valid, y_trn, y_valid

def uniform_y(y_train_orig, y_valid_orig):
    max_log_y = max(np.max(np.log(y_train_orig)), np.max(np.log(y_valid_orig))) * 1.25
    return np.log(y_train_orig)/max_log_y, np.log(y_valid_orig)/max_log_y, max_log_y

def rmsle(y_pred, targ):
    log_vars = np.log(targ + 1) - np.log(y_pred + 1)
    return math.sqrt(np.square(log_vars).mean())

def log_max_inv(preds, mx):
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

def get_bn_model(contin_cols, cat_map_fit):
    contin_inp = Input((contin_cols, ), name='contin')
    contin_out = Dense(
        contin_cols * 10, activation='relu', name='contin_d', kernel_initializer='he_uniform')(contin_inp)
    contin_out = BatchNormalization()(contin_out)
    #contin_out = BatchNormalization()(contin_out)
    embs = [get_emb(feat) for feat in cat_map_fit.features]
    #conts = [get_contin(feat) for feat in contin_map_fit.features]
    #contin_d = [d for inp,d in conts]
    x = Concatenate()([emb for inp, emb in embs] + [contin_out])

    x = Dropout(0.02)(x)
    x = BatchNormalization()(x)
    x = Dense(1000, activation='relu', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Dense(500, activation='relu', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model([inp for inp, emb in embs] + [contin_inp], x)
    model.compile('adam', 'mse')
    return model

def split_cols(arr):
    return np.hsplit(arr,arr.shape[1])


def data2fea(trn, data_dir):
    data = {
        # 'tra': pd.read_csv('{}/air_visit_data.csv'.format(data_dir)),
        # 'tes': pd.read_csv('{}/sample_submission.csv'.format(data_dir)),
        'as': pd.read_csv('{}/air_store_info.csv'.format(data_dir)),
        'hs': pd.read_csv('{}/hpg_store_info.csv'.format(data_dir)),
        'ar': pd.read_csv('{}/air_reserve.csv'.format(data_dir)),
        'hr': pd.read_csv('{}/hpg_reserve.csv'.format(data_dir)),
        'id': pd.read_csv('{}/store_id_relation.csv'.format(data_dir)),
        'hol': pd.read_csv('{}/date_info.csv'.format(data_dir))
    }
    data = get_reserve_tbl(data)
    get_info_from_date(trn, ['visit_date'])
    hol = data["hol"]
    hol.rename(
        {
            'calendar_date': 'Date', 
    }, axis='columns', inplace=True)
    hol.Date = pd.to_datetime(hol.Date)
    fld = 'holiday_flg'
    # get_store_stat_tbl(data)
    hol = add_ts_elapsed(fld, ['af_', 'be_'], hol)
    hol = add_ts_elapsed(fld, ['dur_'], hol)

    # merge everything into store_info
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
    store_info = pd.merge(air_store_info, hpg_store_info, how='left')
    store_info.drop('src', axis=1, inplace=True, errors="ignore")
    store_info[cate_vars] = store_info[cate_vars].fillna('UD')
    store_info[conti_vars] = store_info[conti_vars].fillna(0)


    # from store_info and holiday_info to feature matrix
    mat = trn2mat(trn, store_info, hol, cate_vars, conti_vars)
    cat_map, contin_map, cat_cols, contin_cols, cat_map_fit, y = mat2fea(mat)

    input_map = split_cols(cat_map) + [contin_map]
    feas = {
        'x_map': input_map,
        'y': y,
        'times': trn.visit_date,
        'contin_cols': contin_cols,
        'cat_map_fit': cat_map_fit,
    }
    return feas