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
        trn_like_data = pd.read_csv('{}air_visit_data.csv'.format(PATH))
        test = pd.read_csv('{}sample_submission.csv'.format(PATH))
        test_data = utils.tes2trn(test)
        trn_like_test = test_data.assign(visitors=0)
        result = trn_like_data.groupby('air_store_id').mean().reset_index()
        # add good default for test
        trn_like_test = pd.merge(
            trn_like_test[['air_store_id', 'visit_date']], result)
        dataset = pd.concat([trn_like_data, trn_like_test], axis=0)
        dataset = dataset.reset_index().drop('index', axis=1)
        dataset.to_feather(self.output().path)

    def output(self):
        """
        Returns the target output for this task.
        In this case, a successful execution of this task will create a file in the local file system.
        :return: the target output for this task.
        :rtype: object (:py:class:`luigi.target.Target`)
        """
        return luigi.LocalTarget('{}dataset'.format(RESULT))


class BasicFeas(luigi.Task):
    """
    This task runs over the target data returned by :py:meth:`~/.Streams.output` and
    writes the result into its :py:meth:`~.AggregateArtists.output` target (local file).
    """

    def output(self):
        """
        Returns the target output for this task.
        In this case, a successful execution of this task will create a file on the local filesystem.
        :return: the target output for this task.
        :rtype: object (:py:class:`luigi.target.Target`)
        """
        return luigi.LocalTarget('{}basicfeats'.format(RESULT))

    def requires(self):
        """
        This task's dependencies:
        * :py:class:`~.Streams`
        :return: list of object (:py:class:`luigi.task.Task`)
        """
        return Dataset()

    def run(self):
        dataset = pd.read_feather(self.input().path)
        data = {
            'tra': pd.read_csv('{}air_visit_data.csv'.format(PATH)),
            'as': pd.read_csv('{}air_store_info.csv'.format(PATH)),
            'hs': pd.read_csv('{}hpg_store_info.csv'.format(PATH)),
            'ar': pd.read_csv('{}air_reserve.csv'.format(PATH)),
            'hr': pd.read_csv('{}hpg_reserve.csv'.format(PATH)),
            'id': pd.read_csv('{}store_id_relation.csv'.format(PATH)),
            'hol': pd.read_csv('{}date_info.csv'.format(PATH)),
            'wea': pd.read_csv('{}weather_data_merge.csv'.format(PATH))
        }

        cat_vars = ['air_store_id', 'visit_Year', 'visit_Month',
                    'visit_Week', 'visit_Day', 'visit_Dayofweek', 'visit_Dayofyear',
                    'visit_Is_month_end', 'visit_Is_month_start',
                    'visit_Is_quarter_end', 'visit_Is_quarter_start',
                    'visit_Is_year_end', 'visit_Is_year_start',
                    'visit_Elapsed', 'day_of_week']  # default settings
        contin_vars = []
        utils.add_datepart(dataset, "visit_date", drop=False)
        dataset, cats, contins = utils.add_wea(dataset, data['wea'])
        cat_vars.extend(cats)
        contin_vars.extend(contins)

        dataset, cats, contins = utils.add_holiday_stat(dataset, data['hol'])
        dataset.drop('Date', axis=1, inplace=True, errors='ignore')
        cat_vars.extend(cats)
        contin_vars.extend(contins)

        dataset, cats, contins = utils.add_area_loc_stat(dataset, data)
        cat_vars.extend(cats)
        contin_vars.extend(contins)

        data_statics, _, _ = utils.add_area_loc_stat(data['tra'], data)
        # 'air_store_id', ,  'hpg_loc',
        static_attrs = ['area_name', 'air_loc']
        dataset, cats, contins = utils.add_attr_static(
            dataset, data_statics, static_attrs)
        cat_vars.extend(cats)
        contin_vars.extend(contins)
        dataset.visit_date = pd.to_datetime(dataset.visit_date)
        dataset.visit_date = dataset.visit_date.dt.date

        dataset.to_feather(self.output().path)
        pkl.dump(contin_vars, open(f'{RESULT}basic_contin_vars.pkl', 'wb'))
        pkl.dump(cat_vars, open(f'{RESULT}basic_cat_vars.pkl', 'wb'))


class StoreTsFeas(luigi.Task):
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
        return luigi.LocalTarget('{}_{}_store'.format(RESULT, self.period))

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
