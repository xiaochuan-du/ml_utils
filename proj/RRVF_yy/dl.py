#%%
%matplotlib inline
%reload_ext autoreload
%autoreload 2
from fastai.structured import *
from fastai.column_data import *
import pandas as pd
import keras
import utils
from keras.models import model_from_yaml
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 20
import numpy as np
from pandas_summary import DataFrameSummary
from importlib import reload
reload(utils)

np.set_printoptions(threshold=50, edgeitems=20)

PATH='./data/'

#%%
joined = pd.read_csv(f'{PATH}joined.csv')
joined_test = pd.read_csv(f'{PATH}joined_test.csv')
cat_vars = ['air_store_id','air_genre_name', 'air_area_name']

contin_vars = ['yhat','stores_in_air_area', 'genres_in_air_area',
       'avg_temperature', 'high_temperature', 'low_temperature',
       'precipitation', 'hours_sunlight', 'solar_radiation', 'avg_wind_speed',
       'avg_vapor_pressure', 'avg_local_pressure', 'avg_humidity',
       'avg_sea_pressure', 'cloud_cover' ]
all_vars = cat_vars + contin_vars
dep = 'visitors'
joined = joined[all_vars+[dep, 'visit_date']]
joined_test[dep] = 0
joined_test = joined_test[all_vars+[dep, 'visit_date']]
joined_test['visit_date'] = pd.to_datetime(joined_test.visit_date)
joined['visit_date'] = pd.to_datetime(joined.visit_date)
for v in cat_vars:
    joined[v] = joined[v].astype('category').cat.as_ordered()

apply_cats(joined_test, joined)
for v in contin_vars:
    joined[v] = joined[v].astype('float32')
    joined_test[v] = joined_test[v].astype('float32')

joined = joined.set_index("visit_date")
df, y, nas, mapper = proc_df(joined, 'visitors', do_scale=True)
yl = np.log(y)
joined_test = joined_test.set_index("visit_date")
df_test, _, nas, mapper = proc_df(
    joined_test, 'visitors', do_scale=True, mapper=mapper, na_dict=nas)
val_idx = np.flatnonzero(
    (df.index <= datetime.datetime(2017, 4, 23)) & (df.index >= datetime.datetime(2017, 3, 14)))

#%%
def inv_y(a): return np.exp(a)

def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = np.log1p(targ) - np.log1p(inv_y(y_pred))
    return math.sqrt((pct_var**2).mean())

max_log_y = np.max(yl)
y_range = (0, max_log_y*1.2)

#%%
md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=32,
                                       test_df=df_test)
cat_sz = [(c, len(joined[c].cat.categories)+1) for c in cat_vars]
emb_szs = [(c, min(50, (c+1)//2)) for _, c in cat_sz]


#%%
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.1, 1, [1000, 500, 250], [0.025, 0.25, 0.1], y_range=y_range)
lr = 1e-3
m.lr_find()

#%%
rcParams['figure.figsize'] = 10, 10
m.sched.plot(100)

#%%
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.1, 1, [1000, 500, 250], [0.025, 0.25, 0.1], y_range=y_range)
lr = 1e-3

#%%
m.fit(lr, 10, metrics=[exp_rmspe], cycle_len=1)

#%%
# m.save('dl_l3_bs32')
