" structure data processing "
import pandas as pd


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


def add_ts_elapsed(fld, prefixs, df, dt_var):
    """
        df must be sorted on dt_var,
        and the uniqueness of dt_var will be checked
    """
    if len(df) != len(df[dt_var].unique().tolist()):
        raise ValueError('Please check the uniqueness of {}.'.format(dt_var)) 
    if len(prefixs) == 2:
        # bi-conditions
        prefix = prefixs[0]
        df = df.sort_values([dt_var])
        init_date = df[df[fld] == 1].iloc[0][dt_var]
        sh_el = TimeToEvent(fld, init_date)
        df[prefix + fld] = df.apply(sh_el.get, axis=1).dt.days
        prefix = prefixs[-1]
        df = df.sort_values([dt_var], ascending=[False])
        init_date = df[df[fld] == 1].iloc[0][dt_var]
        sh_el = TimeToEvent(fld, init_date)
        df[prefix + fld] = df.apply(sh_el.get, axis=1).dt.days
        df = df.sort_values([dt_var])
        return df
    else:
        # duration
        prefix = prefixs[0]
        dt_fld = prefix + "time_" + fld
        dur_fld = prefix + fld
        prog_fld = prefix + "prog_" + fld

        df = df.sort_values([dt_var])
        sh_el = DurationTime(fld)
        df[dt_fld] = df.apply(sh_el.get, axis=1)
        prefix = prefixs[0]
        df = df.sort_values([dt_var], ascending=[False])
        sh_el = Duration(dt_fld)
        df[dur_fld] = df.apply(sh_el.get, axis=1)
        df = df.sort_values([dt_var])
        df[prog_fld] = df[dt_fld] / df[dur_fld]
        df[prog_fld].fillna(0, inplace=True)
        return df


def get_info_from_date(data, dt_vars):
    "get_info_from_date"
    data = data.copy()
    for dt_var in dt_vars:
        data[dt_var] = pd.to_datetime(data[dt_var])
        data["{}_week".format(dt_var)] = data[dt_var].dt.week
        data["{}_dayofweek".format(dt_var)] = data[dt_var].dt.dayofweek
        data["{}_year".format(dt_var)] = data[dt_var].dt.year
        data["{}_month".format(dt_var)] = data[dt_var].dt.month
    return data
