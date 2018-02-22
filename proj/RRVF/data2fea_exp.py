# -*- coding: utf-8 -*-
# #
# Copyright 2012-2015 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ts_prep = pickle.load(open('{}ts_prep.pkl'.format(PATH), 'rb'))
# train_set, cats, contins = utils.add_prop(train_set, ts_prep)
# valid_set, *_ = utils.add_prop(valid_set, ts_prep)
# test_set,*_ = utils.add_prop(test_set, ts_prep)
# cat_vars.extend(cats)
# contin_vars.extend(contins)

import re
import random
from functools import reduce
import pickle as pkl
from collections import defaultdict
from heapq import nlargest

from luigi import six

import luigi
import luigi.contrib.hadoop
import luigi.contrib.hdfs
import luigi.contrib.postgres
from utils import proc_df as proc_df

import pandas as pd
import numpy as np
import utils


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
from timeseries_fm import *
from time_series import *
PATH = 'data/' # "../../../data/RRVF/"
RESULT = "result/"


class Dataset(luigi.Task):
    """
    Faked version right now, just generates bogus data.
    """

    def run(self):
        """
        Generates bogus data and writes it into the :py:meth:`~.Streams.output` target.
        """
        data_path = PATH
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
        output_dict = {
            "data": data,
            "hpg_store": hpg_store,
            "air_store": air_store,
            "air_reserve": air_reserve,
            "hpg_reserve": hpg_reserve,
            'date_info': date_info,
        }
        pkl.dump(output_dict, open(self.output().path, 'wb'))

    def output(self):
        """
        Returns the target output for this task.
        In this case, a successful execution of this task will create a file in the local file system.
        :return: the target output for this task.
        :rtype: object (:py:class:`luigi.target.Target`)
        """
        return luigi.LocalTarget('{}dataset.pkl'.format(RESULT))


class LabelSet(luigi.Task):
    """
    label set
    """
    end_date = luigi.Parameter() # static end_date
    ndays = luigi.Parameter() # number of days in labelling set

    def output(self):
        """
        Returns the target output for this task.
        """
        return luigi.LocalTarget(
            '{}label_{}_{}'.format(RESULT, self.end_date, self.ndays))

    def requires(self):
        """
        This task's dependencies:
        * :py:class:`~.Streams`
        :return: list of object (:py:class:`luigi.task.Task`)
        """
        return Dataset()

    def run(self):
        data_dict = pd.read_feather(self.input().path)
        label = get_label(self.end_date, self.ndays, data_dict)
        label.to_feather(self.output().path)

class BaseProc(luigi.Task):
    """
    label set
    """
    end_date = luigi.Parameter() # static end_date
    ndays = luigi.Parameter() # number of days in labelling set
    stat_ndays = luigi.Parameter()

    def get_uri(self):
        class_name = str(self.__class__).split('.')[-1][:-2]
        tmp_str = class_name
        for char in re.findall('[A-Z]', a):
            tmp_str = tmp_str.replace(char, '_'+ char.lower())
        class_name = tmp_str.strip('_')
        return class_name

    def output(self):
        """
        Returns the target output for this task.
        """
        
        return luigi.LocalTarget(
            '{}{}_{}_{}'.format(RESULT, self.get_uri(), self.end_date, self.ndays))

    def requires(self):
        """
        This task's dependencies:
        * :py:class:`~.Streams`
        :return: list of object (:py:class:`luigi.task.Task`)
        """
        return [Dataset(), LabelSet(self.end_date, self.ndays)]

    def run(self):
        data_dict = pkl.load(open(self.input()[0].path, 'rb'))
        label = pd.read_feather(self.input()[1].path)
        key = self.end_date, self.ndays
        func = eval('get_{}_feat'.format(self.get_uri()))
        feas = func(label, key, self.stat_ndays, data_dict)
        feas.to_feather(self.output().path)

class StoreVisitor(BaseProc):
    " one feature engineer processor"
    pass
class StoreExpVisitor(BaseProc):
    " one feature engineer processor"
    pass
class StoreWeek(BaseProc):
    " one feature engineer processor"
    pass
class StoreWeekDiff(BaseProc):
    " one feature engineer processor"
    pass
class StoreAllWeek(BaseProc):
    " one feature engineer processor"
    pass
class StoreWeekExp(BaseProc):
    " one feature engineer processor"
    pass
class StoreHoliday(BaseProc):
    " one feature engineer processor"
    pass
class GenreVisitor(BaseProc):
    " one feature engineer processor"
    pass
class GenreExpVisitor(BaseProc):
    " one feature engineer processor"
    pass
class GenreWeek(BaseProc):
    " one feature engineer processor"
    pass
class GenreWeekExp(BaseProc):
    " one feature engineer processor"
    pass
class FirstLastTime(BaseProc):
    " one feature engineer processor"
    pass
class Reserve(BaseProc):
    " one feature engineer processor"
    pass

class Period(luigi.Task):
    """
    This task runs over the target data returned by :py:meth:`~/.Streams.output` and
    writes the result into its :py:meth:`~.AggregateArtists.output` target (local file).
    """
    end_date = luigi.Parameter() # static end_date
    ndays = luigi.Parameter() # number of days in labelling set
    def output(self):
        """
        Returns the target output for this task.
        In this case, a successful execution of this task will create a file on the local filesystem.
        :return: the target output for this task.
        :rtype: object (:py:class:`luigi.target.Target`)
        """
        return luigi.LocalTarget('{}period'.format(RESULT))

    def requires(self):
        """
        This task's dependencies:
        * :py:class:`~.Streams`
        :return: list of object (:py:class:`luigi.task.Task`)
        """
        return [
            LabelSet(self.end_date, self.ndays),
            StoreVisitor(self.end_date, self.ndays, 1000),
            StoreVisitor(self.end_date, self.ndays, 56),
            StoreVisitor(self.end_date, self.ndays, 28),
            StoreVisitor(self.end_date, self.ndays, 14),
            StoreExpVisitor(self.end_date, self.ndays, 1000),
            StoreWeek(self.end_date, self.ndays, 1000),
            StoreWeek(self.end_date, self.ndays, 56),
            StoreWeek(self.end_date, self.ndays, 28),
            StoreWeek(self.end_date, self.ndays, 14),
            StoreWeekDiff(self.end_date, self.ndays, 58),
            StoreWeekDiff(self.end_date, self.ndays, 1000),
            StoreAllWeek(self.end_date, self.ndays, 1000),
            StoreWeekExp(self.end_date, self.ndays, 1000),
            StoreHoliday(self.end_date, self.ndays, 1000),
            GenreVisitor(self.end_date, self.ndays, 1000),
            GenreVisitor(self.end_date, self.ndays, 56),
            GenreVisitor(self.end_date, self.ndays, 28),
            GenreExpVisitor(self.end_date, self.ndays, 1000),
            GenreWeek(self.end_date, self.ndays, 1000),
            GenreWeek(self.end_date, self.ndays, 56),
            GenreWeek(self.end_date, self.ndays, 28),
            GenreWeekExp(self.end_date, self.ndays, 1000),
            Reserve(self.end_date, self.ndays, -1),
            FirstLastTime(self.end_date, self.ndays, 1000),
        ]

    def run(self):
        datas = [
            pd.read_feather(input_line.path)
            for input_line in self.input()
        ]
        dataset = concat(datas)
        dataset = second_feat(dataset)
        dataset.to_feather(self.output().path)
        # pkl.dump(contin_vars, open(f'{RESULT}basic_contin_vars.pkl', 'wb'))
        # pkl.dump(cat_vars, open(f'{RESULT}basic_cat_vars.pkl', 'wb'))


class StoreTsFeas(luigi.Task):
    """
    This task runs over the target data returned by :py:meth:`~/.Streams.output` and
    writes the result into its :py:meth:`~.AggregateArtists.output` target (local file).
    """
    pivot_date = luigi.Parameter() # edge point between adjustive win and fixed win end_date
    end_date = luigi.Parameter() # end of label set
    date_col = luigi.Parameter()
    date_step = luigi.Parameter()
    days_in_label = luigi.Parameter()
    min_num_in_stat_set = luigi.Parameter()

    def output(self):
        """
        Returns the target output for this task.
        In this case, a successful execution of this task will create a file on the local filesystem.
        :return: the target output for this task.
        :rtype: object (:py:class:`luigi.target.Target`)
        """
        return luigi.LocalTarget('{}_{}_store'.format(RESULT, self.period))

    def requires(self):
        """
        This task's dependencies:
        * :py:class:`~.Streams`
        :return: list of object (:py:class:`luigi.task.Task`)
        """
        return BasicFeas()

    def run(self):
        data = pkl.load(open(self.input().path, 'rb'))

        dataset = pd.read_feather(self.input().path)
        contin_vars = pkl.load(open(f'{RESULT}basic_contin_vars.pkl', 'rb'))
        cat_vars = pkl.load(open(f'{RESULT}basic_cat_vars.pkl', 'rb'))

        dataset, cats, contins = utils.add_rolling_stat(
            dataset, self.period, ['air_store_id'])
        cat_vars.extend(cats)
        contin_vars.extend(contins)

        dataset.visit_date = pd.to_datetime(dataset.visit_date)
        dataset.visit_date = dataset.visit_date.dt.date

        pkl.dump(contin_vars, open(
            '{}_{}_store_contin_vars.pkl'.format(RESULT, self.period), 'wb'))
        pkl.dump(cat_vars, open(
            '{}_{}_store_cat_vars.pkl'.format(RESULT, self.period), 'wb'))
        dataset.to_feather(self.output().path)


class StoreDowTsFeas(luigi.Task):
    """
    This task runs over the target data returned by :py:meth:`~/.Streams.output` and
    writes the result into its :py:meth:`~.AggregateArtists.output` target (local file).
    """
    period = luigi.Parameter()

    def output(self):
        """
        Returns the target output for this task.
        In this case, a successful execution of this task will create a file on the local filesystem.
        :return: the target output for this task.
        :rtype: object (:py:class:`luigi.target.Target`)
        """
        return luigi.LocalTarget('{}_{}_store_dow'.format(RESULT, self.period))

    def requires(self):
        """
        This task's dependencies:
        * :py:class:`~.Streams`
        :return: list of object (:py:class:`luigi.task.Task`)
        """
        return BasicFeas()

    def run(self):
        dataset = pd.read_feather(self.input().path)
        contin_vars = pkl.load(open(f'{RESULT}basic_contin_vars.pkl', 'rb'))
        cat_vars = pkl.load(open(f'{RESULT}basic_cat_vars.pkl', 'rb'))

        dataset, cats, contins = utils.add_rolling_stat(
            dataset, self.period, ['air_store_id', 'visit_Dayofweek'])
        cat_vars.extend(cats)
        contin_vars.extend(contins)

        dataset.visit_date = pd.to_datetime(dataset.visit_date)
        dataset.visit_date = dataset.visit_date.dt.date

        pkl.dump(contin_vars, open(
            '{}_{}_store_dow_contin_vars.pkl'.format(RESULT, self.period), 'wb'))
        pkl.dump(cat_vars, open(
            '{}_{}_store_dow_cat_vars.pkl'.format(RESULT, self.period), 'wb'))
        dataset.to_feather(self.output().path)


class GenreDowTsFeas(luigi.Task):
    """
    This task runs over the target data returned by :py:meth:`~/.Streams.output` and
    writes the result into its :py:meth:`~.AggregateArtists.output` target (local file).
    """
    period = luigi.Parameter()

    def output(self):
        """
        Returns the target output for this task.
        In this case, a successful execution of this task will create a file on the local filesystem.
        :return: the target output for this task.
        :rtype: object (:py:class:`luigi.target.Target`)
        """
        return luigi.LocalTarget('{}_{}_genre_dow'.format(RESULT, self.period))

    def requires(self):
        """
        This task's dependencies:
        * :py:class:`~.Streams`
        :return: list of object (:py:class:`luigi.task.Task`)
        """
        return BasicFeas()

    def run(self):
        dataset = pd.read_feather(self.input().path)
        contin_vars = pkl.load(open(f'{RESULT}basic_contin_vars.pkl', 'rb'))
        cat_vars = pkl.load(open(f'{RESULT}basic_cat_vars.pkl', 'rb'))
        
        dataset, cats, contins = utils.add_rolling_stat(
            dataset, self.period, ['genre_name', 'air_loc', 'visit_Dayofweek'])
        cat_vars.extend(cats)
        contin_vars.extend(contins)

        dataset.visit_date = pd.to_datetime(dataset.visit_date)
        dataset.visit_date = dataset.visit_date.dt.date

        pkl.dump(contin_vars, open(
            '{}_{}_genre_dow_contin_vars.pkl'.format(RESULT, self.period), 'wb'))
        pkl.dump(cat_vars, open(
            '{}_{}_genre_dow_cat_vars.pkl'.format(RESULT, self.period), 'wb'))
        dataset.to_feather(self.output().path)


class AggTsFeas(luigi.Task):
    """
    This task runs over the target data returned by :py:meth:`~/.AggregateArtists.output` or
    :py:meth:`~/.AggregateArtistsHadoop.output` in case :py:attr:`~/.Top10Artists.use_hadoop` is set and
    writes the result into its :py:meth:`~.Top10Artists.output` target (a file in local filesystem).
    """

    def requires(self):
        """
        This task's dependencies:
        * :py:class:`~.AggregateArtists` or
        * :py:class:`~.AggregateArtistsHadoop` if :py:attr:`~/.Top10Artists.use_hadoop` is set.
        :return: object (:py:class:`luigi.task.Task`)
        """
        periods = ['7d', '10d', '14d', '21d', '30d', '60d', '90d', '180d', '360d', '720d']
        return [StoreDowTsFeas(prd) for prd in periods] + [StoreTsFeas(prd) for prd in periods] + [GenreDowTsFeas(prd) for prd in periods]

    def output(self):
        """
        Returns the target output for this task.
        In this case, a successful execution of this task will create a file on the local filesystem.
        :return: the target output for this task.
        :rtype: object (:py:class:`luigi.target.Target`)
        """
        return luigi.LocalTarget('{}agg_feas'.format(RESULT))

    def run(self):
        dfs = []
        contins = []
        cats = []
        for input_file in self.input():
            data_file = input_file.path
            dataset = pd.read_feather(data_file)
            contin_vars = pkl.load(
                open('{}_contin_vars.pkl'.format(data_file), 'rb'))
            cat_vars = pkl.load(
                open('{}_cat_vars.pkl'.format(data_file), 'rb'))
            dfs.append(dataset)
            contins.append(contin_vars)
            cats.append(cat_vars)
        contin_vars = reduce(lambda x, y: list(
            set(x) | set(y)), contins, contins[0])
        cat_vars = reduce(lambda x, y: list(set(x) | set(y)), cats, cats[0])
        cols = [list(df.columns) for df in dfs]
        base_cols = reduce(lambda x, y: list(
            set(x) & set(y)), cols, cols[0])

        dataset = pd.concat(
            [dfs[0][base_cols]] + [df[list(set(df.columns) - set(base_cols))] for df in dfs], axis=1)

        pkl.dump(contin_vars, open(
            '{}_agg_feas_contin_vars.pkl'.format(RESULT), 'wb'))
        pkl.dump(cat_vars, open(
            '{}_agg_feas_cat_vars.pkl'.format(RESULT), 'wb'))
        dataset.to_feather(self.output().path)


def apply_cats(df, trn):
    """Changes any columns of strings in df into categorical variables using trn as
    a template for the category codes.

    Parameters:
    -----------
    df: A pandas dataframe. Any columns of strings will be changed to
        categorical values. The category codes are determined by trn.

    trn: A pandas dataframe. When creating a category for df, it looks up the
        what the category's code were in trn and makes those the category codes
        for df.

    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a

    note the type of col2 is string

    >>> train_cats(df)
    >>> df

       col1 col2
    0     1    a
    1     2    b
    2     3    a

    now the type of col2 is category {a : 1, b : 2}

    >>> df2 = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['b', 'a', 'a']})
    >>> apply_cats(df2, df)

           col1 col2
        0     1    b
        1     2    a
        2     3    a

    now the type of col is category {a : 1, b : 2}
    """
    for n, c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name == 'category'):
            df[n] = pd.Categorical(
                c, categories=trn[n].cat.categories, ordered=True)


def dataset_split(data_raw):
    "dataset_split"
    def split(df):
        trn_len = int(np.floor(len(df) * 0.9))
        valid_len = len(df) - trn_len
        df['type'] = 0  # 0 for train 1 for valid
        indexs = df.index
        df = df.reset_index()
        df.loc[trn_len:, 'type'] = 1
        return df

    test = pd.read_csv('{}sample_submission.csv'.format(PATH))
    test_data = utils.tes2trn(test)
    test_stores = test_data.air_store_id.unique()
    data_raw.visit_date = data_raw.visit_date.astype('str')
    test_data.visit_date = test_data.visit_date.astype('str')
    test_set = data_raw[data_raw.visit_date.isin(
        test_data.visit_date.unique())]
    data_raw = data_raw[~data_raw.visit_date.isin(
        test_data.visit_date.unique())]
    data = data_raw[data_raw.air_store_id.isin(test_stores)]
    tag_data = data.groupby('air_store_id').apply(split)
    t = tag_data.set_index('index')
    train_set = t[t.type == 0]
    valid_set = t[t.type == 1]
    train_set = train_set.reset_index().drop(['index', 'type'], axis=1)
    valid_set = valid_set.reset_index().drop(['index', 'type'], axis=1)
    return train_set, valid_set, test_set


class DataSplits(luigi.Task):
    """
    This task runs over the target data returned by :py:meth:`~/.AggregateArtists.output` or
    :py:meth:`~/.AggregateArtistsHadoop.output` in case :py:attr:`~/.Top10Artists.use_hadoop` is set and
    writes the result into its :py:meth:`~.Top10Artists.output` target (a file in local filesystem).
    """

    def requires(self):
        """
        This task's dependencies:
        * :py:class:`~.AggregateArtists` or
        * :py:class:`~.AggregateArtistsHadoop` if :py:attr:`~/.Top10Artists.use_hadoop` is set.
        :return: object (:py:class:`luigi.task.Task`)
        """
        return AggTsFeas()

    def output(self):
        """
        Returns the target output for this task.
        In this case, a successful execution of this task will create a file on the local filesystem.
        :return: the target output for this task.
        :rtype: object (:py:class:`luigi.target.Target`)
        """
        return luigi.LocalTarget('{}_datasplits.pkl'.format(RESULT))

    def run(self):
        dataset = pd.read_feather(self.input().path)
        contin_vars = pkl.load(
            open(f'{RESULT}_agg_feas_contin_vars.pkl', 'rb'))
        cat_vars = pkl.load(open(f'{RESULT}_agg_feas_cat_vars.pkl', 'rb'))

        train_set, valid_set, test_set = dataset_split(dataset)
        dep = 'visitors'

        n = len(train_set)
        train_set = train_set[cat_vars +
                              contin_vars + [dep, 'visit_date']].copy()
        valid_set = valid_set[cat_vars +
                              contin_vars + [dep, 'visit_date']].copy()
        test_set = test_set[cat_vars +
                            contin_vars + [dep, 'visit_date']].copy()
        for v in cat_vars:
            train_set[v] = train_set[v].astype('category').cat.as_ordered()
        apply_cats(test_set, train_set)
        apply_cats(valid_set, train_set)

        valid_set = valid_set.set_index("visit_date")
        train_set = train_set.set_index("visit_date")
        test_set = test_set.set_index("visit_date")
        to_drop = ['rolling_air_store_id_visit_Dayofweek_14d_skew' ,'rolling_air_store_id_visit_Dayofweek_7d_skew', 'rolling_air_store_id_visit_Dayofweek_10d_skew', 'rolling_genre_name_air_loc_visit_Dayofweek_7d_std', 'rolling_genre_name_air_loc_visit_Dayofweek_10d_skew', 'rolling_air_store_id_visit_Dayofweek_7d_std', 'rolling_genre_name_air_loc_visit_Dayofweek_7d_skew', 'rolling_genre_name_air_loc_visit_Dayofweek_14d_skew']
        train_set.drop(to_drop, axis=1, inplace=True, errors='ignore')
        valid_set.drop(to_drop, axis=1, inplace=True, errors='ignore')
        test_set.drop(to_drop, axis=1, inplace=True, errors='ignore')
        contin_vars = list(set(contin_vars) - set(to_drop))
        cat_vars = list(set(cat_vars) - set(to_drop))
        df, y, nas, mapper = proc_df(train_set, 'visitors', do_scale=True)
        yl = np.log1p(y)
        df_val, y_val, _, _ = proc_df(valid_set, 'visitors', do_scale=True,  # skip_flds=['Id'],
                                      mapper=mapper, na_dict=nas)
        y2 = np.log1p(y_val)
        df_test, _, _, _ = proc_df(test_set, 'visitors', do_scale=True,  # skip_flds=['Id'],
                                   mapper=mapper, na_dict=nas)
        output_dict = {
            'contin_vars': contin_vars,
            'cat_vars': cat_vars,
            'trn': df,
            'trn_y': yl,
            'val': df_val,
            'val_y': y2,
            'test': df_test
        }
        pkl.dump(output_dict, open(
            '{}_datasplits.pkl'.format(RESULT), 'wb'))


if __name__ == "__main__":
    luigi.run()
