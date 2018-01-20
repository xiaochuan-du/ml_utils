#%%
import os
os.chdir('proj/RRVF')
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
DATA_DIR = r'E:\workspace\ai\ml_utils\proj\RRVF\data'

cate_vars = [
    'genre_name',
    'area_name',
    'hpb_area_name',
    'hpb_genre_name',
]
conti_vars = ['latitude', 'longitude', 'hpb_latitude', 'hpb_longitude']

if ENV == 'dev':
    # DEBUG
    import matplotlib.pyplot as plt
    from IPython.display import HTML

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
#%%
for tbl, df in data.items():
    print(tbl)
    display(df.head())

#%%

#%%
test_df = utils.tes2trn(data['tes'])
test_df.visit_date = pd.to_datetime(test_df.visit_date)
display(test_df.head())
display(DataFrameSummary(test_df).summary())

# 

#%%
""" 
test dataset:
2017-04-23 to 2017-05-31
39 unique days in test day
821 air_store_id

trn dataset:
478 unique days
829 stores

store not in test:
'air_0ead98dd07e7a82a',
 'air_229d7e508d9f1b5e',
 'air_2703dcb33192b181',
 'air_b2d8bc9c88b85f96',
 'air_cb083b4789a8d3a2',
 'air_cf22e368c1a71d53',
 'air_d0a7bd3339c3d12a',
 'air_d63cfa6d6ab78446'

it makes sense to use the same time range of trn data as valid set,
hense it tends to have similar ditribution as the test set
[2016-04-23 ~ 2016-06-01]
"""
display(test_df.visit_date.max() - test_df.visit_date.min())
display(len(test_df.visit_date.unique()))
display(len(test_df.air_store_id.unique()))

trn = data['tra']
trn.visit_date = pd.to_datetime(trn.visit_date)

display(trn.visit_date.max() - trn.visit_date.min())
display(len(trn.visit_date.unique()))
display(len(trn.air_store_id.unique()))

display(set(test_df.air_store_id.tolist()) - set(trn.air_store_id.tolist()))

hol = data['hol']
hol.calendar_date = pd.to_datetime(hol.calendar_date)
hol_about_test = hol[(hol.calendar_date >= '2017-05-01') &
                     (hol.calendar_date <= '2017-05-31')]
hol_about_valid = hol[(hol.calendar_date >= '2016-05-01') &
                      (hol.calendar_date <= '2016-05-31')]

display(hol_about_test.head(10))
display(hol_about_test.tail(10))
display(hol_about_valid.head(10))
display(hol_about_valid.tail(10))
# plt.plot(hol_about_test.calendar_date, hol_about_test.holiday_flg)

#%%
display(trn.head())

# y_pred = model.predict(map_valid)
# # rmsle(y_pred, targ = y_valid_orig)
# a = log_max_inv(y_pred, mx=max_log_y)
# diff = a - y_valid_orig
# # print(pred)
#%%
test_df['visitors'] = np.NaN
feas = utils.data2fea(test_df, data_dir)
contin_cols = feas['contin_cols']
cat_map_fit = feas['cat_map_fit']
# valid & trn splitting

model = utils.get_model(contin_cols, cat_map_fit)
#%%
# plt.plot(y_valid_orig)
model.load_weights('./result/caching.h5', by_name=True)

#%%
# display(DataFrameSummary(trn).summary())
pred = model.predict(feas['x_map'])

#%%
import os
# os.chdir(r'E:\workspace\ai\ml_utils\proj\RRVF')
os.chdir(r'/Users/kevindu/Documents/workspace/ml_utils/proj/RRVF')
import glob
import re
import pickle

import numpy as np
import pandas as pd
from isoweek import Week
from pandas_summary import DataFrameSummary
from keras.models import model_from_yaml
import utils
# import xgboost
import random
from importlib import reload
reload(utils)

data_dir = r'./data'
trn = pd.read_csv('{}/air_visit_data.csv'.format(data_dir))
feas = utils.data2fea(trn, 
                        data_dir, 
                        run_para= {
                            "af_etl": 'result/caching.csv'
                        },
                        is_test= False)
print(feas.keys())
tidy_data = feas['tidy_data']
tidy_data.head()

#%%

tidy_data.columns

#  = tes_like_trn.visitors.v# tes_like_trn = tes_like_trn.assign(fill_from=
#     pd.to_datetime(tes_like_trn.visit_date) - pd.Timedelta('365 days'))
# tes_like_trn.fill_from = tes_like_trn.fill_from.astype('str')
# tes_like_trn = pd.merge(tes_like_trn[['air_store_id', 'fill_from']], hist_df, how='left', 
#     left_on=['air_store_id', 'fill_from'], right_on=['air_store_id', 'visit_date'])
# DataFrameSummary(tes_like_trn).summary()

#%%
trn = pd.read_csv('{}/air_visit_data.csv'.format(data_dir))
trn.visit_date = pd.to_datetime(trn.visit_date)
trn.visit_date = trn.visit_date.astype('int') / trn.visit_date.astype('int').max()
trn.visit_date.head()
#%%
start = 40
step = 10
DataFrameSummary(tidy_data[tidy_data.columns[start:start + step]]).summary()

#%%
from importlib import reload
reload(utils)
# from location to area
"""
store_of_same_genre_in_area , store_of_same_genre_in_area
"""
# for grp_key in ['air_loc', 'hpb_loc', 'area_name', 'hpb_area_name']:
#     for genre_key in ['genre_name', 'hpb_genre_name']:
#         var_name = 'stores_of_{}_in_{}'.format(genre_key, grp_key)
#         agg = tidy_data[['air_store_id', grp_key, genre_key]].groupby([grp_key, genre_key])[
#             'air_store_id'].count().reset_index().rename(
#                 {
#                     'air_store_id': var_name,
#                 }, axis='columns')
#         tidy_data = pd.merge(tidy_data, agg, on=[grp_key, genre_key], how='left')
feats = feas['nn_fea']
y = feas['y']
date_sr = pd.to_datetime(tidy_data.Date)
dat = utils.data_split_by_date(feats, y, date_sr, trn2val_ratio=9, step_days=50)
len(dat)
# output = {'trn': [], 'valid': []}
# input_trn = []
# input_valid = []
# y_trn = np.concatenate((y.iloc[:s_i].values, y.iloc[e_i:].values), axis=0)
# y_valid = y.iloc[s_i:e_i]
# for fea in input_map:
#     input_valid.append(fea[s_i:e_i])

#%%
tidy_data.columns


#%%

store_info.head()

#%%
