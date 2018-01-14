#%%
import glob
import re

import numpy as np
import pandas as pd
from isoweek import Week
from pandas_summary import DataFrameSummary
import utils
from importlib import reload
reload(utils)
# %load_ext autoreload
# %autoreload 2

#%%
ENV = 'dev'
DATA_DIR = r'/Users/kevindu/Documents/workspace/ml_utils/proj/RRVF/data'
cate_vars = ['genre_name', 'area_name', 'hpb_area_name', 'hpb_genre_name', ]
conti_vars = ['latitude', 'longitude', 'hpb_latitude', 'hpb_longitude']

if ENV == 'dev':
    # DEBUG
    import matplotlib.pyplot as plt
    from IPython.display import HTML
    # matplotlib inline

if __name__ == '__main__':
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


    data = utils.get_reserve_tbl(data)
    trn = data['tra']
    utils.get_info_from_date(trn, ['visit_date'])
    hol = data["hol"]
    hol.rename(
        {
            'calendar_date': 'Date', 
    }, axis='columns', inplace=True)
    hol.Date = pd.to_datetime(hol.Date)
    fld = 'holiday_flg'
    # get_store_stat_tbl(data)
    hol = utils.add_ts_elapsed(fld, ['af_', 'be_'], hol)
    hol = utils.add_ts_elapsed(fld, ['dur_'], hol)

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
    mat = utils.trn2mat(data['tra'], store_info, hol, cate_vars, conti_vars)
    cat_map, contin_map, cat_cols, contin_cols, cat_map_fit, y = utils.mat2fea(mat)

    # valid & trn splitting
    ts_data = utils.ts_data_split([cat_map, contin_map, y])
    cat_map_train, contin_map_train, y_train_orig = ts_data['trn']
    cat_map_valid, contin_map_valid, y_valid_orig = ts_data['valid']

    y_train, y_valid, max_log_y = utils.uniform_y(y_train_orig, y_valid_orig)

    map_train = utils.split_cols(cat_map_train) + [contin_map_train]
    map_valid = utils.split_cols(cat_map_valid) + [contin_map_valid]
    # map_train = split_cols(cat_map_train) + split_cols(contin_map_train)
    # map_valid = split_cols(cat_map_valid) + split_cols(contin_map_valid)

    # model = utils.get_model(contin_cols, cat_map_fit)
    # model.optimizer.lr = 1e-4
    # hist = model.fit(
    #     map_train,
    #     y_train,
    #     batch_size=128,
    #     epochs=5,
    #     validation_data=(map_valid, y_valid))

    # hist.model.save_weights('./result/caching.h5')
    # model.evaluate(map_valid, y_valid)
