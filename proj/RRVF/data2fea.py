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
        static_attrs = ['area_name', 'air_loc'] # 'air_store_id', ,  'hpg_loc', 
        dataset, cats, contins = utils.add_attr_static(dataset, data_statics, static_attrs)
        cat_vars.extend(cats)
        contin_vars.extend(contins)
        dataset.visit_date = pd.to_datetime(dataset.visit_date)
        dataset.visit_date = dataset.visit_date.dt.date

        dataset.to_feather(self.output().path)
        pkl.dump(contin_vars, open(f'{RESULT}basic_contin_vars.pkl', 'wb'))
        pkl.dump(cat_vars, open(f'{RESULT}basic_cat_vars.pkl','wb'))

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


# class Top10Artists(luigi.Task):
#     """
#     This task runs over the target data returned by :py:meth:`~/.AggregateArtists.output` or
#     :py:meth:`~/.AggregateArtistsHadoop.output` in case :py:attr:`~/.Top10Artists.use_hadoop` is set and
#     writes the result into its :py:meth:`~.Top10Artists.output` target (a file in local filesystem).
#     """

#     date_interval = luigi.DateIntervalParameter()
#     use_hadoop = luigi.BoolParameter()

#     def requires(self):
#         """
#         This task's dependencies:
#         * :py:class:`~.AggregateArtists` or
#         * :py:class:`~.AggregateArtistsHadoop` if :py:attr:`~/.Top10Artists.use_hadoop` is set.
#         :return: object (:py:class:`luigi.task.Task`)
#         """
#         if self.use_hadoop:
#             return AggregateArtistsHadoop(self.date_interval)
#         else:
#             return AggregateArtists(self.date_interval)

#     def output(self):
#         """
#         Returns the target output for this task.
#         In this case, a successful execution of this task will create a file on the local filesystem.
#         :return: the target output for this task.
#         :rtype: object (:py:class:`luigi.target.Target`)
#         """
#         return luigi.LocalTarget("data/top_artists_%s.tsv" % self.date_interval)

#     def run(self):
#         top_10 = nlargest(10, self._input_iterator())
#         with self.output().open('w') as out_file:
#             for streams, artist in top_10:
#                 out_line = '\t'.join([
#                     str(self.date_interval.date_a),
#                     str(self.date_interval.date_b),
#                     artist,
#                     str(streams)
#                 ])
#                 out_file.write((out_line + '\n'))

#     def _input_iterator(self):
#         with self.input().open('r') as in_file:
#             for line in in_file:
#                 artist, streams = line.strip().split()
#                 yield int(streams), artist


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
