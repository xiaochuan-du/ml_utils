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

import pandas as pd
import numpy as np
import utils

PATH = "../../../data/RRVF/"
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
        periods = ['30d', '60d', '90d', '180d', '360d']
        return [StoreDowTsFeas(prd) for prd in periods] + [StoreTsFeas(prd) for prd in periods]

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
        cols = [ list(df.columns) for df in dfs ]
        base_cols = reduce(lambda x, y: list(
            set(x) & set(y)), cols, cols[0])

        dataset = pd.concat(
            [ dfs[0][base_cols] ] + [df[list(set(df.columns) - set(base_cols))] for df in dfs]
            , axis=1)

        pkl.dump(contin_vars, open(
            '{}_agg_feas_contin_vars.pkl'.format(RESULT), 'wb'))
        pkl.dump(cat_vars, open(
            '{}_agg_feas_cat_vars.pkl'.format(RESULT), 'wb'))
        dataset.to_feather(self.output().path)


# class StreamsHdfs(Streams):
#     """
#     This task performs the same work as :py:class:`~.Streams` but its output is written to HDFS.
#     This class uses :py:meth:`~.Streams.run` and
#     overrides :py:meth:`~.Streams.output` so redefine HDFS as its target.
#     """

#     def output(self):
#         """
#         Returns the target output for this task.
#         In this case, a successful execution of this task will create a file in HDFS.
#         :return: the target output for this task.
#         :rtype: object (:py:class:`luigi.target.Target`)
#         """
#         return luigi.contrib.hdfs.HdfsTarget(self.date.strftime('data/streams_%Y_%m_%d_faked.tsv'))


# class AggregateArtistsHadoop(luigi.contrib.hadoop.JobTask):
#     """
#     This task runs a :py:class:`luigi.contrib.hadoop.JobTask` task
#     over each target data returned by :py:meth:`~/.StreamsHdfs.output` and
#     writes the result into its :py:meth:`~.AggregateArtistsHadoop.output` target (a file in HDFS).
#     This class uses :py:meth:`luigi.contrib.spark.SparkJob.run`.
#     """

#     date_interval = luigi.DateIntervalParameter()

#     def output(self):
#         """
#         Returns the target output for this task.
#         In this case, a successful execution of this task will create a file in HDFS.
#         :return: the target output for this task.
#         :rtype: object (:py:class:`luigi.target.Target`)
#         """
#         return luigi.contrib.hdfs.HdfsTarget(
#             "data/artist_streams_%s.tsv" % self.date_interval,
#             format=luigi.contrib.hdfs.PlainDir
#         )

#     def requires(self):
#         """
#         This task's dependencies:
#         * :py:class:`~.StreamsHdfs`
#         :return: list of object (:py:class:`luigi.task.Task`)
#         """
#         return [StreamsHdfs(date) for date in self.date_interval]

#     def mapper(self, line):
#         """
#         The implementation of the map phase of the Hadoop job.
#         :param line: the input.
#         :return: tuple ((key, value) or, in this case, (artist, 1 stream count))
#         """
#         _, artist, _ = line.strip().split()
#         yield artist, 1

#     def reducer(self, key, values):
#         """
#         The implementation of the reducer phase of the Hadoop job.
#         :param key: the artist.
#         :param values: the stream count.
#         :return: tuple (artist, count of streams)
#         """
#         yield key, sum(values)


# class ArtistToplistToDatabase(luigi.contrib.postgres.CopyToTable):
#     """
#     This task runs a :py:class:`luigi.contrib.postgres.CopyToTable` task
#     over the target data returned by :py:meth:`~/.Top10Artists.output` and
#     writes the result into its :py:meth:`~.ArtistToplistToDatabase.output` target which,
#     by default, is :py:class:`luigi.contrib.postgres.PostgresTarget` (a table in PostgreSQL).
#     This class uses :py:meth:`luigi.contrib.postgres.CopyToTable.run`
#     and :py:meth:`luigi.contrib.postgres.CopyToTable.output`.
#     """

#     date_interval = luigi.DateIntervalParameter()
#     use_hadoop = luigi.BoolParameter()

#     host = "localhost"
#     database = "toplists"
#     user = "luigi"
#     password = "abc123"  # ;)
#     table = "top10"

#     columns = [("date_from", "DATE"),
#                ("date_to", "DATE"),
#                ("artist", "TEXT"),
#                ("streams", "INT")]

#     def requires(self):
#         """
#         This task's dependencies:
#         * :py:class:`~.Top10Artists`
#         :return: list of object (:py:class:`luigi.task.Task`)
#         """
#         return Top10Artists(self.date_interval, self.use_hadoop)


if __name__ == "__main__":
    luigi.run()
