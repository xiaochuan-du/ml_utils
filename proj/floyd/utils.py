import pandas as pd
import numpy as np


def add_rolling_stat(dataset, period='60d', grp_keys=['air_store_id']):
    # todo fix test init values
    " add_rolling_stat"
    pre_vars = dataset.columns
    trn = dataset[['visitors', 
                   'visit_date'
                  ]  + grp_keys].copy()
    trn.visit_date = pd.to_datetime(trn.visit_date)

    def func(a_store_df, period=period):
        a_store_df = a_store_df.set_index('visit_date')
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
    print('before groupby')
    trn = trn.groupby(grp_keys).apply(func)
    print('after groupby')
    trn.index = trn.index.droplevel()
    trn = trn.drop('visitors', axis='columns', errors='ignore')
    trn.visit_date = trn.visit_date.astype('str')
    dataset.visit_date = dataset.visit_date.astype('str')
    cols = list(set(trn.columns) - set(grp_keys)) + ['air_store_id']
    trn = pd.merge(dataset, trn[cols], how='left', on=['air_store_id', 'visit_date'])
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
               'hours_sunlight','solar_radiation', 'total_snowfall', 'avg_humidity', ]]
    wea_df = wea.sort_values(['air_store_id', 'visit_date']).fillna(
        method='bfill', axis=1)
    trn.visit_date = trn.visit_date.astype('str')
    wea_df.visit_date = wea_df.visit_date.astype('str')
    en_trn = pd.merge(trn, wea_df, how='left', on=[
                      'visit_date', 'air_store_id'])
    en_trn.visit_date = pd.to_datetime(en_trn.visit_date)

    return en_trn, [], ['avg_temperature',
               'hours_sunlight','solar_radiation', 'total_snowfall', 'avg_humidity']


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
