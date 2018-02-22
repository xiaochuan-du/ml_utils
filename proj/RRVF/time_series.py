"""
    This is a framework to deal with general time series ML.
"""
import time
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import date, timedelta
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
import functools

import arrow
import math
from timeseries_fm import TimeseriesDataset, diff_of_days, date_add_days, left_merge


def get_data(data_path):
    """
    merge
    preprocessing
    """
    air_reserve = pd.read_csv(
        data_path + 'air_reserve.csv').rename(columns={'air_store_id': 'store_id'})
    hpg_reserve = pd.read_csv(
        data_path + 'hpg_reserve.csv').rename(columns={'hpg_store_id': 'store_id'})
    air_store = pd.read_csv(
        data_path + 'air_store_info.csv').rename(columns={'air_store_id': 'store_id'})
    hpg_store = pd.read_csv(
        data_path + 'hpg_store_info.csv').rename(columns={'hpg_store_id': 'store_id'})
    air_visit = pd.read_csv(
        data_path + 'air_visit_data.csv').rename(columns={'air_store_id': 'store_id'})
    store_id_map = pd.read_csv(
        data_path + 'store_id_relation.csv').set_index('hpg_store_id', drop=False)
    date_info = pd.read_csv(data_path + 'date_info.csv').rename(
        columns={'calendar_date': 'visit_date'}).drop('day_of_week', axis=1)
    submission = pd.read_csv(data_path + 'sample_submission.csv')

    submission['visit_date'] = submission['id'].str[-10:]
    submission['store_id'] = submission['id'].str[:-11]
    air_reserve['visit_date'] = air_reserve['visit_datetime'].str[:10]
    air_reserve['reserve_date'] = air_reserve['reserve_datetime'].str[:10]
    air_reserve['dow'] = pd.to_datetime(air_reserve['visit_date']).dt.dayofweek
    hpg_reserve['visit_date'] = hpg_reserve['visit_datetime'].str[:10]
    hpg_reserve['reserve_date'] = hpg_reserve['reserve_datetime'].str[:10]
    hpg_reserve['dow'] = pd.to_datetime(hpg_reserve['visit_date']).dt.dayofweek
    air_visit['id'] = air_visit['store_id'] + '_' + air_visit['visit_date']
    hpg_reserve['store_id'] = hpg_reserve['store_id'].map(
        store_id_map['air_store_id']).fillna(hpg_reserve['store_id'])
    hpg_store['store_id'] = hpg_store['store_id'].map(
        store_id_map['air_store_id']).fillna(hpg_store['store_id'])
    # consider genre in hpg as air genre
    hpg_store.rename(columns={'hpg_genre_name': 'air_genre_name',
                              'hpg_area_name': 'air_area_name'}, inplace=True)
    data = pd.concat([air_visit, submission]).copy()
    data['dow'] = pd.to_datetime(data['visit_date']).dt.dayofweek

    # take weekend 5 6 1, as a kind of holiday
    # dow is a very important feature
    date_info['holiday_flg2'] = pd.to_datetime(
        date_info['visit_date']).dt.dayofweek
    date_info['holiday_flg2'] = ((date_info['holiday_flg2'] > 4) | (
        date_info['holiday_flg'] == 1)).astype(int)

    # Split on area name, should also consider the number of competitors within a distance
    air_store['air_area_name0'] = air_store['air_area_name'].apply(
        lambda x: x.split(' ')[0])
    lbl = LabelEncoder()
    air_store['air_genre_name'] = lbl.fit_transform(
        air_store['air_genre_name'])
    air_store['air_area_name0'] = lbl.fit_transform(
        air_store['air_area_name0'])

    # per the chanllege request
    data['visitors'] = np.log1p(data['visitors'])
    data = data.merge(air_store, on='store_id', how='left')
    data = data.merge(date_info[['visit_date', 'holiday_flg', 'holiday_flg2']], on=[
                      'visit_date'], how='left')
    result = {
        "data": data,
        "hpg_store": hpg_store,
        "air_store": air_store,
        "air_reserve": air_reserve,
        "hpg_reserve": hpg_reserve,
        'date_info': date_info,
    }
    return result


def get_label(end_date, n_day, data_dict):
    """ 
    end_date : end of statistic set
    n_day: the span of label set
    """
    data = data_dict['data']
    date_info = data_dict['date_info']
    label_end_date = date_add_days(end_date, n_day)
    label = data[(data['visit_date'] < label_end_date) &
                 (data['visit_date'] >= end_date)].copy()
    label['end_date'] = end_date
    # diff of pivot date and visit date
    # related to weighting
    label['diff_of_day'] = label['visit_date'].apply(
        lambda x: diff_of_days(x, end_date))
    label['month'] = label['visit_date'].str[5:7].astype(int)
    label['year'] = label['visit_date'].str[:4].astype(int)
    # before & after holiday trend
    for i in [3, 2, 1, -1]:
        date_info_temp = date_info.copy()
        date_info_temp['visit_date'] = date_info_temp['visit_date'].apply(
            lambda x: date_add_days(x, i))
        date_info_temp.rename(columns={'holiday_flg': 'ahead_holiday_{}'.format(
            i), 'holiday_flg2': 'ahead_holiday2_{}'.format(i)}, inplace=True)
        label = label.merge(date_info_temp, on=['visit_date'], how='left')
    label = label.reset_index(drop=True)
    return label


def get_store_visitor_feat(label, key, n_day, data_dict):
    data = data_dict['data']
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (
        data.visit_date > start_date)].copy()
    result = data_temp.groupby(['store_id'], as_index=False)['visitors'].agg({'store_min{}'.format(n_day): 'min',
                                                                              'store_mean{}'.format(n_day): 'mean',
                                                                              'store_median{}'.format(n_day): 'median',
                                                                              'store_max{}'.format(n_day): 'max',
                                                                              'store_count{}'.format(n_day): 'count',
                                                                              'store_std{}'.format(n_day): 'std',
                                                                              'store_skew{}'.format(n_day): 'skew'})
    result = left_merge(label, result, on=['store_id']).fillna(0)
    return result


def get_store_exp_visitor_feat(label, key, n_day, data_dict):
    data = data_dict['data']
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (
        data.visit_date > start_date)].copy()
    data_temp['visit_date'] = data_temp['visit_date'].apply(
        lambda x: diff_of_days(key[0], x))
    data_temp['weight'] = data_temp['visit_date'].apply(lambda x: 0.985**x)
    data_temp['visitors'] = data_temp['visitors'] * data_temp['weight']
    result1 = data_temp.groupby(['store_id'], as_index=False)['visitors'].agg({
        'store_exp_mean{}'.format(n_day): 'sum'})
    result2 = data_temp.groupby(['store_id'], as_index=False)['weight'].agg(
        {'store_exp_weight_sum{}'.format(n_day): 'sum'})
    result = result1.merge(result2, on=['store_id'], how='left')
    result['store_exp_mean{}'.format(n_day)] = result['store_exp_mean{}'.format(
        n_day)] / result['store_exp_weight_sum{}'.format(n_day)]
    result = left_merge(label, result, on=['store_id']).fillna(0)
    return result


def get_store_week_feat(label, key, n_day, data_dict):
    data = data_dict['data']
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (
        data.visit_date > start_date)].copy()
    result = data_temp.groupby(['store_id', 'dow'], as_index=False)['visitors'].agg({'store_dow_min{}'.format(n_day): 'min',
                                                                                     'store_dow_mean{}'.format(n_day): 'mean',
                                                                                     'store_dow_median{}'.format(n_day): 'median',
                                                                                     'store_dow_max{}'.format(n_day): 'max',
                                                                                     'store_dow_count{}'.format(n_day): 'count',
                                                                                     'store_dow_std{}'.format(n_day): 'std',
                                                                                     'store_dow_skew{}'.format(n_day): 'skew'})
    result = left_merge(label, result, on=['store_id', 'dow']).fillna(0)
    return result


def get_store_week_diff_feat(label, key, n_day, data_dict):
    data = data_dict['data']
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (
        data.visit_date > start_date)].copy()
    result = data_temp.set_index(['store_id', 'visit_date'])[
        'visitors'].unstack()
    result = result.diff(axis=1).iloc[:, 1:]
    c = result.columns
    result['store_diff_mean'] = np.abs(result[c]).mean(axis=1)
    result['store_diff_std'] = result[c].std(axis=1)
    result['store_diff_max'] = result[c].max(axis=1)
    result['store_diff_min'] = result[c].min(axis=1)
    result = left_merge(label, result[['store_diff_mean', 'store_diff_std',
                                       'store_diff_max', 'store_diff_min']], on=['store_id']).fillna(0)
    return result


def get_store_all_week_feat(label, key, n_day, data_dict):
    data = data_dict['data']
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (
        data.visit_date > start_date)].copy()
    result_temp = data_temp.groupby(['store_id', 'dow'], as_index=False)['visitors'].agg({'store_dow_mean{}'.format(n_day): 'mean',
                                                                                          'store_dow_median{}'.format(n_day): 'median',
                                                                                          'store_dow_sum{}'.format(n_day): 'max',
                                                                                          'store_dow_count{}'.format(n_day): 'count'})
    result = pd.DataFrame()
    for i in range(7):
        result_sub = result_temp[result_temp['dow'] == i].copy()
        result_sub = result_sub.set_index('store_id')
        result_sub = result_sub.add_prefix(str(i))
        result_sub = left_merge(label, result_sub, on=['store_id']).fillna(0)
        result = pd.concat([result, result_sub], axis=1)
    return result


def get_store_week_exp_feat(label, key, n_day, data_dict):
    data = data_dict['data']
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (
        data.visit_date > start_date)].copy()
    data_temp['visit_date'] = data_temp['visit_date'].apply(
        lambda x: diff_of_days(key[0], x))
    data_temp['visitors2'] = data_temp['visitors']
    result = None
    for i in [0.9, 0.95, 0.97, 0.98, 0.985, 0.99, 0.999, 0.9999]:
        data_temp['weight'] = data_temp['visit_date'].apply(lambda x: i**x)
        data_temp['visitors1'] = data_temp['visitors'] * data_temp['weight']
        data_temp['visitors2'] = data_temp['visitors2'] * data_temp['weight']
        result1 = data_temp.groupby(['store_id', 'dow'], as_index=False)[
            'visitors1'].agg({'store_dow_exp_mean{}_{}'.format(n_day, i): 'sum'})
        result3 = data_temp.groupby(['store_id', 'dow'], as_index=False)[
            'visitors2'].agg({'store_dow_exp_mean2{}_{}'.format(n_day, i): 'sum'})
        result2 = data_temp.groupby(['store_id', 'dow'], as_index=False)['weight'].agg(
            {'store_dow_exp_weight_sum{}_{}'.format(n_day, i): 'sum'})
        result_temp = result1.merge(
            result2, on=['store_id', 'dow'], how='left')
        result_temp = result_temp.merge(
            result3, on=['store_id', 'dow'], how='left')
        result_temp['store_dow_exp_mean{}_{}'.format(n_day, i)] = result_temp['store_dow_exp_mean{}_{}'.format(
            n_day, i)] / result_temp['store_dow_exp_weight_sum{}_{}'.format(n_day, i)]
        result_temp['store_dow_exp_mean2{}_{}'.format(n_day, i)] = result_temp['store_dow_exp_mean2{}_{}'.format(
            n_day, i)] / result_temp['store_dow_exp_weight_sum{}_{}'.format(n_day, i)]
        if result is None:
            result = result_temp
        else:
            result = result.merge(
                result_temp, on=['store_id', 'dow'], how='left')
    result = left_merge(label, result, on=['store_id', 'dow']).fillna(0)
    return result


def get_store_holiday_feat(label, key, n_day, data_dict):
    data = data_dict['data']
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (
        data.visit_date > start_date)].copy()
    result1 = data_temp.groupby(['store_id', 'holiday_flg'], as_index=False)['visitors'].agg(
        {'store_holiday_min{}'.format(n_day): 'min',
         'store_holiday_mean{}'.format(n_day): 'mean',
         'store_holiday_median{}'.format(n_day): 'median',
         'store_holiday_max{}'.format(n_day): 'max',
         'store_holiday_count{}'.format(n_day): 'count',
         'store_holiday_std{}'.format(n_day): 'std',
         'store_holiday_skew{}'.format(n_day): 'skew'})
    result1 = left_merge(label, result1, on=[
                         'store_id', 'holiday_flg']).fillna(0)
    result2 = data_temp.groupby(['store_id', 'holiday_flg2'], as_index=False)['visitors'].agg(
        {'store_holiday2_min{}'.format(n_day): 'min',
         'store_holiday2_mean{}'.format(n_day): 'mean',
         'store_holiday2_median{}'.format(n_day): 'median',
         'store_holiday2_max{}'.format(n_day): 'max',
         'store_holiday2_count{}'.format(n_day): 'count',
         'store_holiday2_std{}'.format(n_day): 'std',
         'store_holiday2_skew{}'.format(n_day): 'skew'})
    result2 = left_merge(label, result2, on=[
                         'store_id', 'holiday_flg2']).fillna(0)
    result = pd.concat([result1, result2], axis=1)
    return result


def get_genre_visitor_feat(label, key, n_day, data_dict):
    data = data_dict['data']
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (
        data.visit_date > start_date)].copy()
    result = data_temp.groupby(['air_genre_name'], as_index=False)['visitors'].agg({'genre_min{}'.format(n_day): 'min',
                                                                                    'genre_mean{}'.format(n_day): 'mean',
                                                                                    'genre_median{}'.format(n_day): 'median',
                                                                                    'genre_max{}'.format(n_day): 'max',
                                                                                    'genre_count{}'.format(n_day): 'count',
                                                                                    'genre_std{}'.format(n_day): 'std',
                                                                                    'genre_skew{}'.format(n_day): 'skew'})
    result = left_merge(label, result, on=['air_genre_name']).fillna(0)
    return result


def get_genre_exp_visitor_feat(label, key, n_day, data_dict):
    data = data_dict['data']
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (
        data.visit_date > start_date)].copy()
    data_temp['visit_date'] = data_temp['visit_date'].apply(
        lambda x: diff_of_days(key[0], x))
    data_temp['weight'] = data_temp['visit_date'].apply(lambda x: 0.985**x)
    data_temp['visitors'] = data_temp['visitors'] * data_temp['weight']
    result1 = data_temp.groupby(['air_genre_name'], as_index=False)[
        'visitors'].agg({'genre_exp_mean{}'.format(n_day): 'sum'})
    result2 = data_temp.groupby(['air_genre_name'], as_index=False)['weight'].agg({
        'genre_exp_weight_sum{}'.format(n_day): 'sum'})
    result = result1.merge(result2, on=['air_genre_name'], how='left')
    result['genre_exp_mean{}'.format(n_day)] = result['genre_exp_mean{}'.format(
        n_day)] / result['genre_exp_weight_sum{}'.format(n_day)]
    result = left_merge(label, result, on=['air_genre_name']).fillna(0)
    return result


def get_genre_week_feat(label, key, n_day, data_dict):
    data = data_dict['data']
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (
        data.visit_date > start_date)].copy()
    result = data_temp.groupby(['air_genre_name', 'dow'], as_index=False)['visitors'].agg({'genre_dow_min{}'.format(n_day): 'min',
                                                                                           'genre_dow_mean{}'.format(n_day): 'mean',
                                                                                           'genre_dow_median{}'.format(n_day): 'median',
                                                                                           'genre_dow_max{}'.format(n_day): 'max',
                                                                                           'genre_dow_count{}'.format(n_day): 'count',
                                                                                           'genre_dow_std{}'.format(n_day): 'std',
                                                                                           'genre_dow_skew{}'.format(n_day): 'skew'})
    result = left_merge(label, result, on=['air_genre_name', 'dow']).fillna(0)
    return result


def get_genre_week_exp_feat(label, key, n_day, data_dict):
    data = data_dict['data']
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (
        data.visit_date > start_date)].copy()
    data_temp['visit_date'] = data_temp['visit_date'].apply(
        lambda x: diff_of_days(key[0], x))
    data_temp['weight'] = data_temp['visit_date'].apply(lambda x: 0.985**x)
    data_temp['visitors'] = data_temp['visitors'] * data_temp['weight']
    result1 = data_temp.groupby(['air_genre_name', 'dow'], as_index=False)[
        'visitors'].agg({'genre_dow_exp_mean{}'.format(n_day): 'sum'})
    result2 = data_temp.groupby(['air_genre_name', 'dow'], as_index=False)[
        'weight'].agg({'genre_dow_exp_weight_sum{}'.format(n_day): 'sum'})
    result = result1.merge(result2, on=['air_genre_name', 'dow'], how='left')
    result['genre_dow_exp_mean{}'.format(n_day)] = result['genre_dow_exp_mean{}'.format(
        n_day)] / result['genre_dow_exp_weight_sum{}'.format(n_day)]
    result = left_merge(label, result, on=['air_genre_name', 'dow']).fillna(0)
    return result


def get_first_last_time(label, key, n_day, data_dict):
    data = data_dict['data']
    start_date = date_add_days(key[0], -n_day)
    data_temp = data[(data.visit_date < key[0]) & (
        data.visit_date > start_date)].copy()
    data_temp = data_temp.sort_values('visit_date')
    result = data_temp.groupby('store_id')['visit_date'].agg({'first_time': lambda x: diff_of_days(key[0], np.min(x)),
                                                              'last_time': lambda x: diff_of_days(key[0], np.max(x)), })
    result = left_merge(label, result, on=['store_id']).fillna(0)
    return result

# air_reserve


def get_reserve_feat(label, key, data_dict):
    data = data_dict['data']
    air_reserve = data_dict['air_reserve']
    hpg_reserve = data_dict['hpg_reserve']
    air_store = data_dict['air_store']

    label_end_date = date_add_days(key[0], key[1])
    air_reserve_temp = air_reserve[(air_reserve.visit_date >= key[0]) &             # key[0] 是'2017-04-23'
                                   # label_end_date 是'2017-05-31'
                                   (air_reserve.visit_date < label_end_date) &
                                   (air_reserve.reserve_date < key[0])].copy()
    air_reserve_temp = air_reserve_temp.merge(
        air_store, on='store_id', how='left')
    air_reserve_temp['diff_time'] = (pd.to_datetime(
        air_reserve['visit_datetime']) - pd.to_datetime(air_reserve['reserve_datetime'])).dt.days
    air_reserve_temp = air_reserve_temp.merge(air_store, on='store_id')
    air_result = air_reserve_temp.groupby(['store_id', 'visit_date'])['reserve_visitors'].agg(
        {'air_reserve_visitors': 'sum',
         'air_reserve_count': 'count'})
    air_store_diff_time_mean = air_reserve_temp.groupby(['store_id', 'visit_date'])['diff_time'].agg(
        {'air_store_diff_time_mean': 'mean'})
    air_diff_time_mean = air_reserve_temp.groupby(['visit_date'])['diff_time'].agg(
        {'air_diff_time_mean': 'mean'})
    air_result = air_result.unstack().fillna(0).stack()
    air_date_result = air_reserve_temp.groupby(['visit_date'])['reserve_visitors'].agg({
        'air_date_visitors': 'sum',
        'air_date_count': 'count'})
    hpg_reserve_temp = hpg_reserve[(hpg_reserve.visit_date >= key[0]) & (
        hpg_reserve.visit_date < label_end_date) & (hpg_reserve.reserve_date < key[0])].copy()
    hpg_reserve_temp['diff_time'] = (pd.to_datetime(
        hpg_reserve['visit_datetime']) - pd.to_datetime(hpg_reserve['reserve_datetime'])).dt.days
    hpg_result = hpg_reserve_temp.groupby(['store_id', 'visit_date'])['reserve_visitors'].agg({'hpg_reserve_visitors': 'sum',
                                                                                               'hpg_reserve_count': 'count'})
    hpg_result = hpg_result.unstack().fillna(0).stack()
    hpg_date_result = hpg_reserve_temp.groupby(['visit_date'])['reserve_visitors'].agg({
        'hpg_date_visitors': 'sum',
        'hpg_date_count': 'count'})
    hpg_store_diff_time_mean = hpg_reserve_temp.groupby(['store_id', 'visit_date'])['diff_time'].agg(
        {'hpg_store_diff_time_mean': 'mean'})
    hpg_diff_time_mean = hpg_reserve_temp.groupby(['visit_date'])['diff_time'].agg(
        {'hpg_diff_time_mean': 'mean'})
    air_result = left_merge(label, air_result, on=[
                            'store_id', 'visit_date']).fillna(0)
    air_store_diff_time_mean = left_merge(label, air_store_diff_time_mean, on=[
                                          'store_id', 'visit_date']).fillna(0)
    hpg_result = left_merge(label, hpg_result, on=[
                            'store_id', 'visit_date']).fillna(0)
    hpg_store_diff_time_mean = left_merge(label, hpg_store_diff_time_mean, on=[
                                          'store_id', 'visit_date']).fillna(0)
    air_date_result = left_merge(label, air_date_result, on=[
                                 'visit_date']).fillna(0)
    air_diff_time_mean = left_merge(
        label, air_diff_time_mean, on=['visit_date']).fillna(0)
    hpg_date_result = left_merge(label, hpg_date_result, on=[
                                 'visit_date']).fillna(0)
    hpg_diff_time_mean = left_merge(
        label, hpg_diff_time_mean, on=['visit_date']).fillna(0)
    result = pd.concat([air_result, hpg_result, air_date_result, hpg_date_result, air_store_diff_time_mean,
                        hpg_store_diff_time_mean, air_diff_time_mean, hpg_diff_time_mean], axis=1)
    return result

# second feature


def second_feat(result):
    result['store_mean_14_28_rate'] = result['store_mean14'] / \
        (result['store_mean28'] + 0.01)
    result['store_mean_28_56_rate'] = result['store_mean28'] / \
        (result['store_mean56'] + 0.01)
    result['store_mean_56_1000_rate'] = result['store_mean56'] / \
        (result['store_mean1000'] + 0.01)
    result['genre_mean_28_56_rate'] = result['genre_mean28'] / \
        (result['genre_mean56'] + 0.01)
    result['sgenre_mean_56_1000_rate'] = result['genre_mean56'] / \
        (result['genre_mean1000'] + 0.01)
    return result


if __name__ == '__main__':
    PATH = r'/Users/kevindu/Documents/workspace/ai/data/RRVF/'
    # r'E:\workspace\ai\ml_utils\proj\RRVF\data\\'
    data_dict = get_data(PATH)

    fes = [
        lambda label, key, data_dict: get_store_visitor_feat(label, key, 1000, data_dict),
        lambda label, key, data_dict: get_store_visitor_feat(label, key, 56, data_dict),
        lambda label, key, data_dict: get_store_visitor_feat(label, key, 28, data_dict),
        lambda label, key, data_dict: get_store_visitor_feat(label, key, 14, data_dict),
        lambda label, key, data_dict: get_store_exp_visitor_feat(label, key, 1000, data_dict),
        lambda label, key, data_dict: get_store_week_feat(label, key, 1000, data_dict),
        lambda label, key, data_dict: get_store_week_feat(label, key, 56, data_dict),
        lambda label, key, data_dict: get_store_week_feat(label, key, 28, data_dict),
        lambda label, key, data_dict: get_store_week_feat(label, key, 14, data_dict),
        lambda label, key, data_dict: get_store_week_diff_feat(label, key, 58, data_dict),
        lambda label, key, data_dict: get_store_week_diff_feat(label, key, 1000, data_dict),
        lambda label, key, data_dict: get_store_all_week_feat(label, key, 1000, data_dict),
        lambda label, key, data_dict: get_store_week_exp_feat(label, key, 1000, data_dict),
        lambda label, key, data_dict: get_store_holiday_feat(label, key, 1000, data_dict),
        lambda label, key, data_dict: get_genre_visitor_feat(label, key, 1000, data_dict),
        lambda label, key, data_dict: get_genre_visitor_feat(label, key, 56, data_dict),
        lambda label, key, data_dict: get_genre_visitor_feat(label, key, 28, data_dict),
        lambda label, key, data_dict: get_genre_exp_visitor_feat(label, key, 1000, data_dict),
        lambda label, key, data_dict: get_genre_week_feat(label, key, 1000, data_dict),
        lambda label, key, data_dict: get_genre_week_feat(label, key, 56, data_dict),
        lambda label, key, data_dict: get_genre_week_feat(label, key, 28, data_dict),
        lambda label, key, data_dict: get_reserve_feat(label, key, data_dict),
        lambda label, key, data_dict: get_genre_week_exp_feat(label, key, 1000, data_dict),
        lambda label, key, data_dict: get_first_last_time(label, key, 1000, data_dict)
    ]
    pivot_date='2016-03-12'
    end_date='2016-04-22'
    ts_data = TimeseriesDataset(
        pivot_date=pivot_date,
        end_date=end_date,
        data_dict=data_dict,
        date_col='visit_date',
        date_step=7,
        days_in_label=39,
        min_num_in_stat_set=37,
        label_getter=get_label,
        fes=fes,
        high_eng=second_feat
    )
    dataset = ts_data.get_trn(4)
    fea_path = "pivot_date_{}_end_date_{}".format(
        pivot_date,
        end_date
    )
    dataset.reset_index().to_feather(PATH+fea_path)
