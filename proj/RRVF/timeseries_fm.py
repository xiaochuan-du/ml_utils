"""
    This is a framework to deal with general time series ML.
"""
import math
import time
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import date, timedelta
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
import functools

import arrow

def concat(L):
    "concat"
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            try:
                result[l.columns.tolist()] = l
            except:
                print(l.head())
    return result


def left_merge(data1, data2, on):
    if type(on) != list:
        on = [on]
    if (set(on) & set(data2.columns)) != set(on):
        data2_temp = data2.reset_index()
    else:
        data2_temp = data2.copy()
    columns = [f for f in data2.columns if f not in on]
    result = data1.merge(data2_temp, on=on, how='left')
    result = result[columns]
    return result


def diff_of_days(day1, day2):
    days = (parse(day1[:10]) - parse(day2[:10])).days
    return days


def date_add_days(start_date, days):
    end_date = parse(start_date[:10]) + timedelta(days=days)
    end_date = end_date.strftime('%Y-%m-%d')
    return end_date


def make_feats(data_dict,
               win,
               label_getter,
               fes=[],
               high_eng=None):
    pivot_date = win['pivot_date']
    days_in_label = win['days_in_label']
    key = pivot_date, days_in_label  # the idx of label set
    label = label_getter(pivot_date, days_in_label, data_dict)

    result = [label]
    for feature_eng in fes:
        result.append(feature_eng(label, key, data_dict))
    result.append(label)
    result = concat(result)
    if high_eng:
        result = high_eng(result)
    return result


class TimeseriesDataset():
    def __init__(
            self,
            pivot_date,
            end_date,
            data_dict,
            date_col,
            date_step,
            days_in_label,
            min_num_in_stat_set,
            label_getter,
            fes=[],
            high_eng=None
    ):
        self.__pivot_date = pivot_date
        self.__data_dict = data_dict
        self.__date_col = date_col
        self.__end_date = end_date
        self.__date_step = date_step
        self.__days_in_label = days_in_label
        merged_data = data_dict['data']
        windows = []
        max_date = arrow.get(pivot_date)
        min_date = arrow.get(merged_data.visit_date.min())
        delta = (max_date - min_date).days - min_num_in_stat_set
        nwindows_bf_pivot = int((delta) / date_step)
        nwindows_af_pivot = math.floor(
            (arrow.get(end_date) - arrow.get(pivot_date)).days / date_step)

        start_date = min_date.shift(days=min_num_in_stat_set)
        for day_delta in range(nwindows_bf_pivot):
            # >= start & < end
            windows.append(
                {
                    "pivot_date": start_date.format('YYYY-MM-DD'),
                    "days_in_label": days_in_label
                }
            )
            start_date = start_date.shift(days=date_step)
        start_date = max_date.shift(days=date_step)
        ndays_unit_af_pivot = days_in_label - days_in_label % date_step
        for day_delta in range(nwindows_af_pivot):
            adaptive_len = ndays_unit_af_pivot - (day_delta * date_step)
            windows.append(
                {
                    "pivot_date": start_date.format('YYYY-MM-DD'),
                    "days_in_label": adaptive_len
                }
            )
            start_date = start_date.shift(days=date_step)
        self.__windows = windows
        print('nwindows_bf_pivot:{}, nwindows_af_pivot {}'
              .format(nwindows_bf_pivot, nwindows_af_pivot))
        print('First window {}'.format(windows[0]))
        print('Last window {}'.format(windows[-1]))
        self.__label_getter = label_getter
        self.__fes = fes
        self.__high_eng = high_eng

    def get_trn(self, concurrency=2):
        " get dataframe for trn "
        feats = []
        # step_task = int(len(self.__windows) / 100)
        # num_tasks = len(self.__windows)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            feats = executor.map(
                lambda x: make_feats(win=x,
                                     data_dict=self.__data_dict,
                                     label_getter=self.__label_getter,
                                     fes=self.__fes,
                                     high_eng=self.__high_eng),
                self.__windows)
            print('Done')
        train_feat = pd.concat(feats)
        return train_feat

    def get_test(self, start_date, ndays):
        " get dataframe for test "
        test_feat = make_feats(win={
            "pivot_date": start_date,
            "days_in_label": ndays
        },
            data_dict=self.__data_dict,
            label_getter=self.__label_getter,
            fes=self.__fes,
            high_eng=self.__high_eng)
        return test_feat

    def generate_trn(self):
        pass