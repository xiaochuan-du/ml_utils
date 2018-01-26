#%%
import glob
import re
import pickle

import numpy as np
import pandas as pd
from isoweek import Week
from pandas_summary import DataFrameSummary
import itertools
import utils
import xgboost
import xgboost as xgb
import random
import matplotlib.pyplot as plt
import xgboost
import operator
import random
import pickle
import math
from pandas_summary import DataFrameSummary
from importlib import reload
reload(utils)
import matplotlib
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (20['0, 10['0)

def rmsle(y_pred, targ):
    log_vars = np.log(targ + 1) - np.log(y_pred + 1)
    return math.sqrt(np.square(log_vars).mean())

def plot_impt(model):
    importance = model.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    df.plot(kind='barh', x='feature', y='fscore',
            legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    # df[df.fscore < 0['009].feature.tolist()

def rmsle_wo_log(y_predicted, y_true):
    y_true = y_true.get_label()
    y_predicted_orig = utils.log_max_inv(y_predicted, max_log_y)
    y_true_orig = utils.log_max_inv(y_true, max_log_y)
    score = rmsle(y_predicted_orig, y_true_orig)
    return ('rmsle', score)

def rmsle_wo_log_sk(y_predicted, y_true):
    y_predicted_orig = utils.log_max_inv(y_predicted, max_log_y)
    y_true_orig = utils.log_max_inv(y_true, max_log_y)
    score = rmsle(y_predicted_orig, y_true_orig)
    return score

def split(df):
    trn_len = int(np.floor(len(df) * 0['9))
    valid_len = len(df) - trn_len
    df['type'] = 0  # 0 for train 1 for valid
    indexs = df.index
    df = df.reset_index()
    df.loc[trn_len:, 'type'] = 1
    return df

#%%
prop = pd.read_pickle('./prop.pkl')
#%%
data_raw = pd.read_csv('./data/air_visit_data.csv')
test = pd.read_csv('./data/sample_submission.csv')
#%%
drop_vars = [
    #     'air_store_id',
    'mean_visits',
    #     'air_loc',
    'visit_date_dayofweek',
    #     'area_name',
    #     'genre_name',
    'visit_date_week',
    'visit_date_month',
    #     'hpb_area_name',
    #     'hpb_loc',
    'visit_date_year',
    'holiday_flg'

    # TBD
    'max_visits',
    'std_visits',
    'min_visits',
    'hpb_genre_name',

    'be_holiday_flg',
    'af_holiday_flg',
    'dur_time_holiday_flg',
    'dur_holiday_flg',
    'dur_prog_holiday_flg',

    'rolling_60d_max',
    'rolling_60d_min',
    'rolling_60d_median',
    'rolling_60d_std',

    #     'stores_in_area_name',
    #     'max_visits_in_area_name',
    #     'mean_visits_in_area_name',
    #     'std_visits_in_area_name',
    #     'min_visits_in_area_name',

    #     'stores_in_air_loc',
    #     'mean_visits_in_air_loc',
    #     'std_visits_in_air_loc',
    #     'max_visits_in_air_loc',
    #     'min_visits_in_air_loc',

    # 'visit_date_ts',
    'stores_in_hpb_area_name',
    'std_visits_in_hpb_area_name',
    'max_visits_in_hpb_area_name',
    'min_visits_in_hpb_area_name',
    'mean_visits_in_hpb_area_name',

    'stores_in_hpb_loc',
    'std_visits_in_hpb_loc',
    'max_visits_in_hpb_loc',
    'min_visits_in_hpb_loc',
    'mean_visits_in_hpb_loc',

    'type'
]

 
#%%
def getPropPrediction(df):
    global count
    m_data = df.copy().rename({
        'visit_date': 'ds'
    }, axis='columns')
    store_id = df.air_store_id.unique()[0]
    m = prop[store_id]
    forecast = m.predict(m_data)
    df['yhat'] = np.exp(forecast.yhat.values)
    df['yhat_upper'] = np.exp(forecast.yhat_upper.values)
    df['yhat_lower'] = np.exp(forecast.yhat_lower.values)
    df['trend'] = np.exp(forecast.trend.values)
    df['trend_upper'] = np.exp(forecast.trend_upper.values)
    df['trend_lower'] = np.exp(forecast.trend_lower.values)
    df['1'] = np.exp(forecast['1'].values)
    df['1_upper'] = np.exp(forecast['1_upper'].values)
    df['1_lower'] = np.exp(forecast['1_lower'].values)
    df['2'] = np.exp(forecast['2'].values)
    df['2_upper'] = np.exp(forecast['2_upper'].values)
    df['2_lower'] = np.exp(forecast['2_lower'].values)
    df['3'] = np.exp(forecast['3'].values)
    df['3_upper'] = np.exp(forecast['3_upper'].values)
    df['3_lower'] = np.exp(forecast['3_lower'].values)
    df['6'] = np.exp(forecast['6'].values)
    df['6_upper'] = np.exp(forecast['6_upper'].values)
    df['6_lower'] = np.exp(forecast['6_lower'].values)
    df['holidays'] = np.exp(forecast.holidays.values)
    df['holidays_upper'] = np.exp(forecast.holidays_upper.values)
    df['holidays_lower'] = np.exp(forecast.holidays_lower.values)
    df['seasonal'] = np.exp(forecast.seasonal.values)
    df['seasonal_upper'] = np.exp(forecast.seasonal_upper.values)
    df['seasonal_lower'] = np.exp(forecast.seasonal_lower.values)
    df['seasonalities'] = np.exp(forecast.seasonalities.values)
    df['seasonalities_upper'] = np.exp(forecast.seasonalities_upper.values)
    df['seasonalities_lower'] = np.exp(forecast.seasonalities_lower.values)
    df['weekly'] = np.exp(forecast.weekly.values)
    df['weekly_upper'] = np.exp(forecast.weekly_upper.values)
    df['weekly_lower'] = np.exp(forecast.weekly_lower.values)
    display(str(count) + ' : ' + store_id)
    count = count + 1
    return df

#%%
count = 1
data_with_prop = data_raw.groupby('air_store_id').apply(getPropPrediction)

#%%
data_with_prop.to_csv('./data/air_visit_data_with_prop_feas.csv')

#%%
data_dir = r'./data'
test = pd.read_csv('{}/sample_submission.csv'.format(data_dir))
trn_like_test = utils.tes2trn(test)
count = 1
test_with_prop = trn_like_test.groupby('air_store_id').apply(getPropPrediction)

#%%
test_with_prop.to_csv('./data/submission_data_with_prop.csv')

#%%
test_data = utils.tes2trn(test)
test_stores = test_data.air_store_id.unique()
data = data_with_prop[data_with_prop.air_store_id.isin(test_stores)]
tag_data = data.groupby('air_store_id').apply(split)
t = tag_data.set_index('index')
train_set = t[t.type == 0]
valid_set = t[t.type == 1]
len(train_set), len(train_set.air_store_id.unique()), len(
    valid_set), len(valid_set.air_store_id.unique())


#%%
# train_feas = utils.data2fea(train_set, './data', drop_vars=drop_vars)
valid_feas = utils.data2fea(valid_set, './data', drop_vars=drop_vars)

#%%
train_set.head()

