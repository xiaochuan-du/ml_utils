import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
import math
from pandas_summary import DataFrameSummary

from keras.layers import Concatenate, Dense, Dropout, Embedding, Flatten, Input, BatchNormalization
from keras import initializers
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam

FORCE_CAT = [
    'dur_time_holiday_flg', 'visit_date_week', 'visit_date_dayofweek',
    'visit_date_month', 'dur_holiday_flg', 'dur_prog_holiday_flg', 'air_loc',
    'hpb_loc'
]
FORCE_Y = ['visitors']


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
        return row.Date - self.last


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
        df[prefix + fld] = df.apply(sh_el.get, axis=1).dt.days

        prefix = prefixs[-1]
        df = df.sort_values(['Date'], ascending=[False])
        init_date = df[df[fld] == 1].iloc[0]['Date']
        sh_el = TimeToEvent(fld, init_date)
        df[prefix + fld] = df.apply(sh_el.get, axis=1).dt.days
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
    tes = tes.assign(
        air_store_id=tes["id"].map(lambda x: '_'.join(x.split('_')[:-1])))
    tes = tes.assign(visit_date=tes["id"].map(lambda x: x.split('_')[2]))
    return tes[["air_store_id", "visit_date"]]


def trn2test(tes_in_trn):
    tes_in_trn['id'] = df[['air_store_id', 'visit_date']].apply(
        lambda x: '_'.join(x), axis=1)
    return tes_in_trn[["id"]]


def get_reserve_tbl(data):
    " get_reserve_tbl "
    hpg_reserve = data['hr']
    hpg_store_info = data['hs']
    store_id_relation = data['id']
    air_reserve = data['ar']
    air_store_info = data['as']
    date_info = data['hol']

    hpg_joined = pd.merge(
        hpg_reserve,
        hpg_store_info,
        how='left', )
    hpg_fl_joined = pd.merge(
        store_id_relation,
        hpg_joined,
        how='left', )
    hpg_fl_joined.rename(
        {
            'hpb_latitude': 'latitude',
            'hpb_longitude': 'longitude',
            'hpg_genre_name': 'genre_name',
            'hpg_area_name': 'area_name',
        },
        axis='columns',
        inplace=True)
    hpg_fl_joined.drop('hpg_store_id', axis=1, inplace=True, errors="ignore")
    hpg_fl_joined = hpg_fl_joined.assign(src='hpg')

    air_joined = pd.merge(
        air_reserve,
        air_store_info,
        how='left', )
    air_fl_joined = air_joined
    air_fl_joined.rename(
        {
            'air_genre_name': 'genre_name',
            'air_area_name': 'area_name',
        },
        axis='columns',
        inplace=True)
    air_fl_joined = air_fl_joined.assign(src='air')

    # clean reserve
    reserve = pd.concat([air_fl_joined, hpg_fl_joined], axis=0)
    reserve.visit_datetime = pd.to_datetime(reserve.visit_datetime)
    reserve.reserve_datetime = pd.to_datetime(reserve.reserve_datetime)

    date_info.drop('day_of_week', axis=1, inplace=True, errors="ignore")
    date_info.calendar_date = date_info.calendar_date.astype('str')
    reserve_en = reserve.assign(visit_date=reserve.visit_datetime.dt.date)
    reserve_en.visit_date = reserve_en.visit_date.astype('str')
    reserve_en = pd.merge(
        reserve_en,
        date_info,
        how='left',
        left_on=['visit_date'],
        right_on=['calendar_date'])
    reserve_en.rename(
        {
            'holiday_flg': 'visit_holiday_flg',
        }, axis='columns', inplace=True)
    reserve_en.drop('calendar_date', axis=1, inplace=True, errors="ignore")

    reserve_en = reserve_en.assign(
        reserve_date=reserve_en.reserve_datetime.dt.date)
    reserve_en.reserve_date = reserve_en.reserve_date.astype('str')
    reserve_en = pd.merge(
        reserve_en,
        date_info,
        how='left',
        left_on=['reserve_date'],
        right_on=['calendar_date'])
    reserve_en.rename(
        {
            'holiday_flg': 'reserve_holiday_flg',
        }, axis='columns', inplace=True)
    reserve_en.drop('calendar_date', axis=1, inplace=True, errors="ignore")
    data['reserve'] = reserve_en
    return data


def inspect_var_type(data, force_cat=FORCE_CAT, force_y=FORCE_Y):
    " from dataframe to var types"

    summary_df = DataFrameSummary(data).summary()
    # auto evaluate datatype
    var_types = {
        dtype: [
            col for col in summary_df.columns
            if summary_df.loc["types"][col] == dtype
        ]
        for dtype in ['numeric', 'bool', 'categorical', 'date', 'constant']
    }
    guess_cat_vars = var_types['bool'] + var_types['categorical'] + \
        var_types['date'] + var_types['constant'] + force_cat
    guess_cat_vars = list(set(guess_cat_vars))
    guess_contin_vars = list(
        set(data.columns) - set(guess_cat_vars) - set(force_y))
    # cat_vars = [
    #     'genre_name', 'area_name', 'hpb_genre_name', 'hpb_area_name',
    #     'holiday_flg', 'dur_time_holiday_flg', 'visit_date_week',
    #     'visit_date_dayofweek', 'visit_date_year', 'visit_date_month',
    #     'air_store_id'
    # ]
    # contin_vars = [
    #     'latitude', 'longitude', 'hpb_latitude', 'hpb_longitude',
    #     'af_holiday_flg', 'be_holiday_flg', 'dur_holiday_flg',
    #     'dur_prog_holiday_flg', 'min_visits', 'max_visits', 'mean_visits',
    #     'std_visits'
    # ]
    return guess_cat_vars, guess_contin_vars


def mat2fea(mat, cat_vars, contin_vars):
    " from feature dataframe to matrix based on type"
    mat[contin_vars] = mat[contin_vars].astype('float')
    for v in contin_vars:
        mat.loc[mat[v].isnull(), v] = 0
    for v in cat_vars:
        mat.loc[mat[v].isnull(), v] = ""
    cat_maps = [(o, LabelEncoder()) for o in cat_vars]
    contin_maps = [([o], StandardScaler()) for o in contin_vars]

    cat_mapper = DataFrameMapper(cat_maps)
    cat_map_fit = cat_mapper.fit(mat)
    cat_cols = len(cat_map_fit.features)

    contin_mapper = DataFrameMapper(contin_maps)
    contin_map_fit = contin_mapper.fit(mat)
    contin_cols = len(contin_map_fit.features)

    cat_map = cat_map_fit.transform(mat).astype(np.int64)
    contin_map = contin_map_fit.transform(mat).astype(np.float)

    return cat_map, contin_map, cat_cols, \
        contin_cols, cat_map_fit, mat.visitors


def ts_data_split(input_map, y, s_i, e_i):
    output = {'trn': [], 'valid': []}
    # train_ratio = 0.9
    # size = y.shape[0]
    # trn_size = int(train_ratio * size)
    input_trn = []
    input_valid = []
    y_trn = np.concatenate((y.iloc[:s_i].values, y.iloc[e_i:].values), axis=0)
    y_valid = y.iloc[s_i:e_i]
    for fea in input_map:
        input_valid.append(fea[s_i:e_i])
        input_trn.append(np.concatenate((fea[:s_i], fea[e_i:]), axis=0))

    return input_trn, input_valid, y_trn, y_valid


def data_split_by_date(feats, y, date_sr, trn2val_ratio=9, step_days=50):
    " according to date, split data"
    data_blocks = []
    delta = pd.to_timedelta('{} days'.format(step_days))
    curdt = date_sr.min()
    while curdt < date_sr.max():
        data_index = date_sr[(date_sr >= curdt) &
                             (date_sr < curdt + delta)].index.tolist()
        trn_size = int(trn2val_ratio / 10.0 * len(data_index))
        trn_index = data_index[:trn_size]
        valid_index = data_index[trn_size:]
        y_trn = y[pd.Index(trn_index)]
        y_valid = y[pd.Index(valid_index)]
        if isinstance(feats, list):
            x_trn = []
            x_valid = []
            for feat in feats:
                x_trn.append(feat[pd.Index(trn_index)])
                x_valid.append(feat[pd.Index(valid_index)])
        else:
            x_trn = feats[pd.Index(trn_index)]
            x_valid = feats[pd.Index(valid_index)]
        data_blocks.append({
            "x_trn": x_trn,
            "y_trn": y_trn,
            "x_valid": x_valid,
            "y_valid": y_valid,
        })
        curdt = curdt + delta
    return data_blocks


def uniform_y(y_train_orig, y_valid_orig):
    max_log_y = max(
        np.max(np.log1p(y_train_orig)), np.max(np.log1p(y_valid_orig))) * 1.25
    return np.log1p(y_train_orig) / max_log_y, np.log1p(
        y_valid_orig) / max_log_y, max_log_y


def rmsle(y_pred, targ):
    log_vars = np.log(targ + 1) - np.log(y_pred + 1)
    return math.sqrt(np.square(log_vars).mean())


def log_max_inv(preds, mx):
    return np.exp(preds * mx) - 1


def my_init(scale):
    return lambda shape, name=None: initializations.uniform()


def emb_init(shape, name=None):
    return initializers.RandomUniform()


def cat_map_info(feat):
    return feat[0], len(feat[1].classes_)


def get_emb(feat):
    name, c = cat_map_info(feat)
    c2 = (c + 1) // 2
    if c2 > 50:
        c2 = 50
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


def root_mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def get_bn_model(contin_cols, cat_map_fit):
    contin_inp = Input((contin_cols, ), name='contin')
    contin_out = Dense(
        contin_cols * 10,
        activation='relu',
        name='contin_d',
        kernel_initializer='he_uniform')(contin_inp)
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
    model.compile(Adam(decay=1e-6), loss='mse')
    return model


def split_cols(arr):
    return np.hsplit(arr, arr.shape[1])


def add_rolling_stat(trn, period='60d'):
    # todo fix test init values
    " add_rolling_stat"
    trn_org = trn.copy()
    trn.visit_date = pd.to_datetime(trn.visit_date)

    def func(a_store_df, period=period):
        a_store_df = a_store_df.set_index('visit_date')
        rolling_max = a_store_df["visitors"].rolling(period).max().shift(1)
        rolling_min = a_store_df["visitors"].rolling(period).min().shift(1)
        rolling_median = a_store_df["visitors"].rolling(
            period).median().shift(1)
        rolling_std = a_store_df["visitors"].rolling(period).std().shift(1)
        a_store_df = a_store_df.reset_index()
        for stat, var_name in zip([rolling_max, rolling_min, rolling_median, rolling_std],
                                  ["rolling_{}_max".format(period), "rolling_{}_min".format(period),
                                   "rolling_{}_median".format(period), "rolling_{}_std".format(period)]):
            stat = pd.DataFrame(stat).reset_index()
            stat = stat.rename(
                {
                    'visitors': var_name,
                }, axis="columns"
            )
            a_store_df = pd.merge(
                a_store_df, stat, on='visit_date', how='left')
        return a_store_df
    trn = trn.groupby('air_store_id').apply(func)
    trn.index = trn.index.droplevel()
    trn = trn.drop('visitors', axis='columns', errors='ignore')
    trn.visit_date = trn.visit_date.astype('str')
    trn = pd.merge(trn_org, trn, how='left', on=['air_store_id', 'visit_date'])
    return trn


def add_area_loc_stat(tidy_df, data):
    """
        merge store info in two systems, and generate area /loc
    """
    # store information related features
    data = get_reserve_tbl(data)
    store_info = data["reserve"][[
        'air_store_id', "src", 'genre_name', 'area_name', 'latitude',
        'longitude'
    ]]
    store_info = store_info.drop_duplicates()
    air_store_info = store_info[store_info.src == 'air']
    hpg_store_info = store_info[(store_info.src == 'hpg') &
                                (~store_info.genre_name.isna())]
    hpg_store_info = hpg_store_info.rename(
        {
            'latitude': 'hpb_latitude',
            'longitude': 'hpb_longitude',
            'genre_name': 'hpb_genre_name',
            'area_name': 'hpb_area_name',
        },
        axis='columns')
    hpg_store_info = hpg_store_info.drop('src', axis=1, errors="ignore")
    store_info = pd.merge(air_store_info, hpg_store_info, how='left')
    store_info = store_info.drop('src', axis=1, errors="ignore")

    # region by location
    for loc_vars in [['latitude', 'longitude'],
                     ['hpb_latitude', 'hpb_longitude']]:
        if len(loc_vars[0].split('_')) > 1:
            src = loc_vars[0].split('_')[0]
        else:
            src = 'air'
        store_info[loc_vars] = store_info[loc_vars].astype('str')
        store_info['{}_loc'.format(src)] = store_info[loc_vars].apply(
            lambda x: '_'.join(x), axis=1)
        store_info = store_info.drop(loc_vars, axis=1, errors='ignore')

    # stores' number in region/ area
    for grp_key in ['air_loc', 'hpb_loc', 'area_name', 'hpb_area_name']:
        var_name = 'stores_in_{}'.format(grp_key)
        agg = store_info.groupby(grp_key)['air_store_id'].count()
        agg = pd.DataFrame(agg, columns=[agg.name])
        agg = agg.reset_index()
        agg = agg.rename(
            {
                'air_store_id': var_name,
            }, axis="columns")
        store_info = pd.merge(store_info, agg, on=grp_key, how='left')

    tidy_df = pd.merge(tidy_df, store_info, how='left', on='air_store_id')
    return tidy_df


def add_holiday_stat(tidy_df, hol):
    # holiday related features
    hol = hol.rename(
        {
            'calendar_date': 'Date',
        }, axis='columns')
    hol.Date = pd.to_datetime(hol.Date)
    fld = 'holiday_flg'
    hol = add_ts_elapsed(fld, ['af_', 'be_'], hol)
    hol = add_ts_elapsed(fld, ['dur_'], hol)
    str_date_hol = hol
    str_date_hol.Date = str_date_hol.Date.astype('str')
    tidy_df = pd.merge(
        tidy_df,
        str_date_hol,
        how='left',
        left_on='visit_date',
        right_on='Date')
    return tidy_df


def add_attr_static(tidy_df, attrs):
    "add_attr_static"
    # region/ area's statis
    for key in attrs:
        agg = tidy_df.groupby(key)['visitors'].agg(
            [np.min, np.max, np.mean, np.std]).rename(
                columns={
                    'amin': 'min_{}_in_{}'.format('visits', key),
                    'amax': 'max_{}_in_{}'.format('visits', key),
                    'mean': 'mean_{}_in_{}'.format('visits', key),
                    'std': 'std_{}_in_{}'.format('visits', key)
                })
        agg.reset_index(inplace=True)
        tidy_df = pd.merge(tidy_df, agg, how='left', on=key)
    return tidy_df


def add_default_2_tst(tes_like_trn, trn):
    "add_default_2_tst"
    hist_df = add_rolling_stat(trn, period='60d')

    def func(a_store_df):
        a_store_df = a_store_df.sort_values("visit_date")
        return a_store_df.iloc[-1]
    agg = hist_df.groupby('air_store_id').apply(func)
    agg = agg.drop('air_store_id', errors='ignore',
                   axis='columns').reset_index()
    agg = agg[['air_store_id', 'rolling_60d_median']]
    agg = agg.rename(
        {
            "rolling_60d_median": 'visitors',
        }, axis="columns"
    )
    tes_like_trn = pd.merge(tes_like_trn, agg, how='left')
    tes_like_trn.visitors = tes_like_trn.visitors.values + np.random.rand(tes_like_trn.visitors.values.shape[0])
    return tes_like_trn


def data2fea(src_df, data_dir, run_para={}, is_test=False, drop_vars=None):
    " data cleansing and enrichment"
    af_etl = run_para.get("af_etl", None)
    use_cacheing = run_para.get("use_cacheing", None)
    if not use_cacheing:
        # load data from disk into memory
        data = {
            'tra': pd.read_csv('{}/air_visit_data.csv'.format(data_dir)),
            # 'tes': pd.read_csv('{}/sample_submission.csv'.format(data_dir)),
            'as': pd.read_csv('{}/air_store_info.csv'.format(data_dir)),
            'hs': pd.read_csv('{}/hpg_store_info.csv'.format(data_dir)),
            'ar': pd.read_csv('{}/air_reserve.csv'.format(data_dir)),
            'hr': pd.read_csv('{}/hpg_reserve.csv'.format(data_dir)),
            'id': pd.read_csv('{}/store_id_relation.csv'.format(data_dir)),
            'hol': pd.read_csv('{}/date_info.csv'.format(data_dir))
        }
        if is_test:
            # fill visits
            src_df = add_default_2_tst(src_df, data['tra'])
        tidy_df = add_rolling_stat(src_df)
        tidy_df = add_area_loc_stat(tidy_df, data)
        tidy_df = add_holiday_stat(tidy_df, data["hol"])
        static_attrs = ['air_store_id', 'air_loc',
                        'hpb_loc', 'area_name', 'hpb_area_name']
        tidy_df = add_attr_static(tidy_df, static_attrs)

        # fill datetime splitted data
        get_info_from_date(tidy_df, ['visit_date'])
        # sort data according to their date
        tidy_df = tidy_df.sort_values('visit_date')
        tidy_df = tidy_df.assign(visit_date_ts=tidy_df.visit_date.astype('int') ) 
        mat = tidy_df.drop(['visit_date', 'Date'], axis=1)
    else:
        mat = pd.read_csv(use_cacheing)
    if af_etl:
        mat.to_csv(af_etl,index=False)
    cat_vars, contin_vars = inspect_var_type(mat)
    # fill NaN and drop useless columns
    mat[cat_vars] = mat[cat_vars].fillna('UD')
    mat[contin_vars] = mat[contin_vars].fillna(0)

    if drop_vars:
        mat = mat.drop(drop_vars, axis='columns', errors='ignore')
        cat_vars = list(set(cat_vars) - set(drop_vars))
        contin_vars = list(set(contin_vars) - set(drop_vars))

    # reorder columns
    mat = mat.reindex(sorted(mat.columns), axis=1)

    cat_map, contin_map, cat_cols, contin_cols, cat_map_fit, y = mat2fea(
        mat, cat_vars, contin_vars)

    feas = {
        'nn_fea': split_cols(cat_map) + [contin_map],
        'sk_fea': np.concatenate([cat_map, contin_map], axis=1),
        'y': y,
        'contin_cols': contin_cols,
        'cat_map_fit': cat_map_fit,
        'tidy_data': mat,
        'all_vars': cat_vars + contin_vars
    }
    return feas
