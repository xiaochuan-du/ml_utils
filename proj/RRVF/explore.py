
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
test_df = utils.tes2trn(data['tes'])
display(test_df.head())
# y_pred = model.predict(map_valid)
# # rmsle(y_pred, targ = y_valid_orig)
# a = log_max_inv(y_pred, mx=max_log_y)
# diff = a - y_valid_orig
# # print(pred)

# #%%
# plt.plot(y_valid_orig)

# #%%
# display(DataFrameSummary(trn).summary())

