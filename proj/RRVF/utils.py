import re
import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype, is_numeric_dtype
import warnings
import sklearn
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from sklearn_pandas import DataFrameMapper


def add_datepart(df, fldname, drop=True):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.

    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.

    Examples:
    ---------

    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df

        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13

    >>> add_datepart(df, 'A')
    >>> df

        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed
    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800
    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200
    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600
    """
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
              'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop:
        df.drop(fldname, axis=1, inplace=True)


def add_rolling_stat(dataset, period='60d', grp_keys=['air_store_id']):
    # todo fix test init values
    " add_rolling_stat"
    pre_vars = dataset.columns
    trn = dataset[['visitors',
                   'visit_date'
                   ] + grp_keys].copy()
    trn.visit_date = pd.to_datetime(trn.visit_date)

    def func(a_store_df, period=period):
        a_store_df = a_store_df.groupby(['visit_date']).mean()
#         a_store_df = a_store_df.set_index('visit_date')
        a_store_df = a_store_df.sort_index()
        # print(a_store_df["visitors"].rolling(period))
        rolling_max = a_store_df["visitors"].rolling(period).max().shift(1)
        rolling_min = a_store_df["visitors"].rolling(period).min().shift(1)
        rolling_median = a_store_df["visitors"].rolling(
            period).median().shift(1)
        rolling_std = a_store_df["visitors"].rolling(period).std().shift(1)
        rolling_cnt = a_store_df["visitors"].rolling(period).count().shift(1)
        rolling_mean = a_store_df["visitors"].rolling(period).mean().shift(1)
        rolling_skew = a_store_df["visitors"].rolling(period).skew().shift(1)

        a_store_df = a_store_df.reset_index()
        for stat, var_name in zip([
            rolling_max,
            rolling_min,
            rolling_median,
            rolling_std,
            rolling_cnt,
            rolling_mean,
            rolling_skew],
            [
            "rolling_{}_{}_max".format('_'.join(grp_keys), period),
            "rolling_{}_{}_min".format('_'.join(grp_keys), period),
            "rolling_{}_{}_median".format('_'.join(grp_keys), period),
            "rolling_{}_{}_std".format('_'.join(grp_keys), period),
            "rolling_{}_{}_cnt".format('_'.join(grp_keys), period),
            "rolling_{}_{}_mean".format('_'.join(grp_keys), period),
                "rolling_{}_{}_skew".format('_'.join(grp_keys), period)]):
            stat = pd.DataFrame(stat).reset_index()
            stat = stat.rename(
                {
                    'visitors': var_name,
                }, axis="columns"
            )
            a_store_df = pd.merge(
                a_store_df, stat, on='visit_date', how='left')
        return a_store_df
    trn = trn.groupby(grp_keys).apply(func)
#     display(trn.head())
    trn = trn.drop(grp_keys, axis='columns', errors='ignore').reset_index(level=grp_keys)
#     trn.index = trn.index.droplevel()
    trn = trn.drop('visitors', axis='columns', errors='ignore')
    trn.visit_date = trn.visit_date.astype('str')
    dataset.visit_date = dataset.visit_date.astype('str')
    # display(trn)
    cols = list(set(trn.columns)) #  - set(grp_keys)) + ['air_store_id']
    trn = pd.merge(dataset, trn[cols], how='left',
                   on=['visit_date']+grp_keys )
#     display(trn.head())
    return trn, [], list(set(trn.columns) - set(pre_vars))


def tes2trn(tes):
    tes = tes.assign(
        air_store_id=tes["id"].map(lambda x: '_'.join(x.split('_')[:-1])))
    tes = tes.assign(visit_date=tes["id"].map(lambda x: x.split('_')[2]))
    return tes[["air_store_id", "visit_date"]]


def add_prop(trn, ts_feat):
    ' add prophet features to train like dataframe'
    for model_key in ts_feat.keys():
        ts_feat[model_key] = ts_feat[model_key].assign(air_store_id=model_key)
        ts_feat[model_key] = ts_feat[model_key].rename({
            'ds': 'visit_date',
        }, axis="columns")
    concated = pd.concat([res for res in ts_feat.values()], axis=0)
    concated.yhat = np.exp(concated.yhat.values)
    concated.visit_date = concated.visit_date.astype('str')
    name_dict = {col: 'prop_{}'.format(
        col) for col in concated.columns if col != 'visit_date' and col != 'air_store_id'}
    concated.rename(columns=dict(name_dict), inplace=True)
    trn.visit_date = trn.visit_date.astype('str')
    en_trn = pd.merge(trn, concated[['prop_yhat_lower', 'prop_yhat_upper', 'prop_yhat',
                                     'visit_date', 'air_store_id']], on=['visit_date', 'air_store_id'], how='left')
    return en_trn, [], ['prop_yhat_lower', 'prop_yhat_upper', 'prop_yhat']


def add_wea(trn, wea):
    ' add prophet features to train like dataframe'
    wea = wea[['visit_date', 'air_store_id', 'avg_temperature',
               'hours_sunlight', 'solar_radiation', 'total_snowfall', 'avg_humidity', ]]
    wea_df = wea.sort_values(['air_store_id', 'visit_date']).fillna(
        method='bfill', axis=1)
    trn.visit_date = trn.visit_date.astype('str')
    wea_df.visit_date = wea_df.visit_date.astype('str')
    en_trn = pd.merge(trn, wea_df, how='left', on=[
                      'visit_date', 'air_store_id'])
    en_trn.visit_date = pd.to_datetime(en_trn.visit_date)

    return en_trn, [], ['avg_temperature',
                        'hours_sunlight', 'solar_radiation', 'total_snowfall', 'avg_humidity']


def trn2test(tes_in_trn):
    tes_in_trn['id'] = tes_in_trn[['air_store_id', 'visit_date']].apply(
        lambda x: '_'.join(x), axis=1)
    return tes_in_trn[["id", 'visitors']]


def add_holiday_stat(tidy_df, hol):
    # holiday related features
    hol = hol.rename(
        {
            'calendar_date': 'Date',
        }, axis='columns')
    hol.Date = pd.to_datetime(hol.Date)
    fld = 'holiday_flg'
    hol = add_ts_elapsed(fld, ['af_', 'be_'], hol)
    hol = add_ts_elapsed(fld, ['dur_'], hol)
    str_date_hol = hol
    str_date_hol.Date = str_date_hol.Date.astype('str')
    tidy_df.visit_date = tidy_df.visit_date.astype('str')
    tidy_df = pd.merge(
        tidy_df,
        str_date_hol,
        how='left',
        left_on='visit_date',
        right_on='Date')
    tidy_df.visit_date = pd.to_datetime(tidy_df.visit_date)
    return tidy_df, ['holiday_flg', 'af_holiday_flg',
                     'be_holiday_flg', 'dur_time_holiday_flg', 'dur_holiday_flg',
                     'dur_prog_holiday_flg', ], []


class TimeToEvent(object):
    'iter across row'

    def __init__(self, fld, init_date):
        '__init__'
        self.fld = fld
        self.last = init_date

    def get(self, row):
        'getter'
        if (row[self.fld] == 1):
            self.last = row.Date
        return row.Date - self.last


class DurationTime(object):
    'iter across row'

    def __init__(self, fld):
        '__init__'
        self.fld = fld
        self.dura = 0

    def get(self, row):
        'getter'
        if (row[self.fld] == 1):
            self.dura = self.dura + 1
        else:
            self.dura = 0
        return self.dura


class Duration(object):
    'iter across row'

    def __init__(self, fld):
        self.fld = fld
        self.dura = 0
        self.past = 0

    def get(self, row):
        if (row[self.fld] != 0 and self.past != 0):
            self.dura = self.dura
        elif (row[self.fld] != 0 and self.past == 0):
            self.dura = row[self.fld]
        else:
            self.dura = 0
        self.past = row[self.fld]
        return self.dura


def add_ts_elapsed(fld, prefixs, df):
    "add_elapsed"
    if len(prefixs) == 2:
        # bi-conditions
        prefix = prefixs[0]
        df = df.sort_values(['Date'])
        init_date = df[df[fld] == 1].iloc[0]['Date']
        sh_el = TimeToEvent(fld, init_date)
        df[prefix + fld] = df.apply(sh_el.get, axis=1).dt.days

        prefix = prefixs[-1]
        df = df.sort_values(['Date'], ascending=[False])
        init_date = df[df[fld] == 1].iloc[0]['Date']
        sh_el = TimeToEvent(fld, init_date)
        df[prefix + fld] = df.apply(sh_el.get, axis=1).dt.days
        df = df.sort_values(['Date'])
        return df
    else:
        # duration
        prefix = prefixs[0]
        dt_fld = prefix + "time_" + fld
        dur_fld = prefix + fld
        prog_fld = prefix + "prog_" + fld

        df = df.sort_values(['Date'])
        sh_el = DurationTime(fld)
        df[dt_fld] = df.apply(sh_el.get, axis=1)
        prefix = prefixs[0]
        df = df.sort_values(['Date'], ascending=[False])
        sh_el = Duration(dt_fld)
        df[dur_fld] = df.apply(sh_el.get, axis=1)
        df = df.sort_values(['Date'])
        df[prog_fld] = df[dt_fld] / df[dur_fld]
        df[prog_fld].fillna(0, inplace=True)
        return df


def get_reserve_tbl(data):
    " get_reserve_tbl "
    hpg_reserve = data['hr']
    hpg_store_info = data['hs']
    store_id_relation = data['id']
    air_reserve = data['ar']
    air_store_info = data['as']
    date_info = data['hol']

    hpg_joined = pd.merge(
        hpg_reserve,
        hpg_store_info,
        how='left', )
    hpg_fl_joined = pd.merge(
        store_id_relation,
        hpg_joined,
        how='left', )
    hpg_fl_joined.rename(
        {
            'hpb_latitude': 'latitude',
            'hpb_longitude': 'longitude',
            'hpg_genre_name': 'genre_name',
            'hpg_area_name': 'area_name',
        },
        axis='columns',
        inplace=True)
    hpg_fl_joined.drop('hpg_store_id', axis=1, inplace=True, errors="ignore")
    hpg_fl_joined = hpg_fl_joined.assign(src='hpg')

    air_joined = pd.merge(
        air_reserve,
        air_store_info,
        how='left', )
    air_fl_joined = air_joined
    air_fl_joined.rename(
        {
            'air_genre_name': 'genre_name',
            'air_area_name': 'area_name',
        },
        axis='columns',
        inplace=True)
    air_fl_joined = air_fl_joined.assign(src='air')

    # clean reserve
    reserve = pd.concat([air_fl_joined, hpg_fl_joined], axis=0)
    reserve.visit_datetime = pd.to_datetime(reserve.visit_datetime)
    reserve.reserve_datetime = pd.to_datetime(reserve.reserve_datetime)

    date_info.drop('day_of_week', axis=1, inplace=True, errors="ignore")
    date_info.calendar_date = date_info.calendar_date.astype('str')
    reserve_en = reserve.assign(visit_date=reserve.visit_datetime.dt.date)
    reserve_en.visit_date = reserve_en.visit_date.astype('str')
    reserve_en = pd.merge(
        reserve_en,
        date_info,
        how='left',
        left_on=['visit_date'],
        right_on=['calendar_date'])
    reserve_en.rename(
        {
            'holiday_flg': 'visit_holiday_flg',
        }, axis='columns', inplace=True)
    reserve_en.drop('calendar_date', axis=1, inplace=True, errors="ignore")

    reserve_en = reserve_en.assign(
        reserve_date=reserve_en.reserve_datetime.dt.date)
    reserve_en.reserve_date = reserve_en.reserve_date.astype('str')
    reserve_en = pd.merge(
        reserve_en,
        date_info,
        how='left',
        left_on=['reserve_date'],
        right_on=['calendar_date'])
    reserve_en.rename(
        {
            'holiday_flg': 'reserve_holiday_flg',
        }, axis='columns', inplace=True)
    reserve_en.drop('calendar_date', axis=1, inplace=True, errors="ignore")
    data['reserve'] = reserve_en
    return data


def add_area_loc_stat(tidy_df, data):
    """
        merge store info in two systems, and generate area /loc
    """
    # store information related features
    hpg_store_info = data['hs']
    store_id_relation = data['id']
    air_store_info = data['as']

    hpg_joined = hpg_store_info
    hpg_joined.rename(
        {
            'latitude': 'hpg_latitude',
            'longitude': 'hpg_longitude',
        },
        axis='columns',
        inplace=True)

    hpg_fl_joined = pd.merge(
        store_id_relation,
        hpg_joined,
        how='left', )
    hpg_fl_joined.drop('hpg_store_id', axis=1, inplace=True, errors="ignore")

    store_info = pd.merge(air_store_info, hpg_fl_joined, how='left')
    store_info.rename(
        {
            'air_genre_name': 'genre_name',
            'air_area_name': 'area_name',
        },
        axis='columns',
        inplace=True)

    # region by location
    for loc_vars in [['latitude', 'longitude'],
                     ['hpg_latitude', 'hpg_longitude']]:
        if len(loc_vars[0].split('_')) > 1:
            src = loc_vars[0].split('_')[0]
        else:
            src = 'air'
        store_info[loc_vars] = store_info[loc_vars].astype('str')
        store_info['{}_loc'.format(src)] = store_info[loc_vars].apply(
            lambda x: '_'.join(x), axis=1)
        store_info = store_info.drop(loc_vars, axis=1, errors='ignore')
    # stores' number in region/ area
    for grp_key in ['air_loc', 'hpg_loc', 'area_name', 'hpg_area_name']:
        var_name = 'stores_in_{}'.format(grp_key)
        agg = store_info.groupby(grp_key)['air_store_id'].count()
        agg = pd.DataFrame(agg, columns=[agg.name])
        agg = agg.reset_index()
        agg = agg.rename(
            {
                'air_store_id': var_name,
            }, axis="columns")
        store_info = pd.merge(store_info, agg, on=grp_key, how='left')

    tidy_df = pd.merge(tidy_df, store_info, how='left', on='air_store_id')
    return tidy_df, ['genre_name', 'area_name', 'hpg_genre_name',
                     'hpg_area_name', 'air_loc', 'hpg_loc'], ['stores_in_air_loc',
                                                              'stores_in_hpg_loc', 'stores_in_area_name', 'stores_in_hpg_area_name', ]


def add_attr_static(tidy_df, data_statics, attrs):
    "add_attr_static"
    # region/ area's statis
    contins = []
    pre_vars = tidy_df.columns
    for key in attrs:
        agg = data_statics.groupby(key)['visitors'].agg(
            [np.min, np.max, np.mean, np.std]).rename(
                columns={
                    'amin': 'min_{}_in_{}'.format('visits', key),
                    'amax': 'max_{}_in_{}'.format('visits', key),
                    'mean': 'mean_{}_in_{}'.format('visits', key),
                    'std': 'std_{}_in_{}'.format('visits', key)
                })
        agg.reset_index(inplace=True)
        contins.extend(agg.columns)
        tidy_df = pd.merge(tidy_df, agg[[
            'min_{}_in_{}'.format('visits', key),
            'max_{}_in_{}'.format('visits', key),
            'mean_{}_in_{}'.format('visits', key),
            'std_{}_in_{}'.format('visits', key),
            key
        ]], how='left', on=key)

    return tidy_df, [], list(set(contins) - set(pre_vars))


def fix_missing(df, col, name, na_dict):
    """ Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing.

    Parameters:
    -----------
    df: The data frame that will be changed.

    col: The column of data to fix by filling in missing data.

    name: The name of the new filled column in df.

    na_dict: A dictionary of values to create na's of and the value to insert. If
        name is not a key of na_dict the median will fill any missing data. Also
        if name is not a key of na_dict and there is no missing data in col, then
        no {name}_na column is not created.


    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2

    >>> fix_missing(df, df['col1'], 'col1', {})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1     2    2    True
    2     3    2   False


    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2

    >>> fix_missing(df, df['col2'], 'col2', {})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2


    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2

    >>> fix_missing(df, df['col1'], 'col1', {'col1' : 500})
    >>> df
       col1 col2 col1_na
    0     1    5   False
    1   500    2    True
    2     3    2   False
    """
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

def numericalize(df, col, name, max_n_cat):
    """ Changes the column col from a categorical type to it's integer codes.

    Parameters:
    -----------
    df: A pandas dataframe. df[name] will be filled with the integer codes from
        col.

    col: The column you wish to change into the categories.
    name: The column name you wish to insert into df. This column will hold the
        integer codes.

    max_n_cat: If col has more categories than max_n_cat it will not change the
        it to its integer codes. If max_n_cat is None, then col will always be
        converted.

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

    now the type of col2 is category { a : 1, b : 2}

    >>> numericalize(df, df['col2'], 'col3', None)

       col1 col2 col3
    0     1    a    1
    1     2    b    2
    2     3    a    1
    """
    if not is_numeric_dtype(col) and ( max_n_cat is None or col.nunique()>max_n_cat):
        df[name] = col.cat.codes+1

def scale_vars(df, mapper):
    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper

def proc_df(df, y_fld, skip_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):

    """ proc_df takes a data frame df and splits off the response variable, and
    changes the df into an entirely numeric dataframe.

    Parameters:
    -----------
    df: The data frame you wish to process.

    y_fld: The name of the response variable

    skip_flds: A list of fields that dropped from df.

    do_scale: Standardizes each column in df,Takes Boolean Values(True,False)

    na_dict: a dictionary of na columns to add. Na columns are also added if there
        are any missing values.

    preproc_fn: A function that gets applied to df.

    max_n_cat: The maximum number of categories to break into dummy values, instead
        of integer codes.

    subset: Takes a random subset of size subset from df.

    mapper: If do_scale is set as True, the mapper variable
        calculates the values used for scaling of variables during training time(mean and standard deviation).

    Returns:
    --------
    [x, y, nas, mapper(optional)]:

        x: x is the transformed version of df. x will not have the response variable
            and is entirely numeric.

        y: y is the response variable

        nas: returns a dictionary of which nas it created, and the associated median.

        mapper: A DataFrameMapper which stores the mean and standard deviation of the corresponding continous
        variables which is then used for scaling of during test-time.

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

    now the type of col2 is category { a : 1, b : 2}

    >>> x, y, nas = proc_df(df, 'col1')
    >>> x

       col2
    0     1
    1     2
    2     1

    >>> data = DataFrame(pet=["cat", "dog", "dog", "fish", "cat", "dog", "cat", "fish"],
                 children=[4., 6, 3, 3, 2, 3, 5, 4],
                 salary=[90, 24, 44, 27, 32, 59, 36, 27])

    >>> mapper = DataFrameMapper([(:pet, LabelBinarizer()),
                          ([:children], StandardScaler())])

    >>>round(fit_transform!(mapper, copy(data)), 2)

    8x4 Array{Float64,2}:
    1.0  0.0  0.0   0.21
    0.0  1.0  0.0   1.88
    0.0  1.0  0.0  -0.63
    0.0  0.0  1.0  -0.63
    1.0  0.0  0.0  -1.46
    0.0  1.0  0.0  -0.63
    1.0  0.0  0.0   1.04
    0.0  0.0  1.0   0.21
    """
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    df = df.copy()
    if preproc_fn: preproc_fn(df)
    y = df[y_fld].values
    df.drop(skip_flds+[y_fld], axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    res = [pd.get_dummies(df, dummy_na=True), y, na_dict]
    if do_scale: res = res + [mapper]
    return res

