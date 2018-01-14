import numpy as np
import pandas as pd

class TimeToEvent(object):
    'iter across row'
    def __init__(self, fld, init_date):
        self.fld = fld
        self.last = init_date
        # self.last_store = 0
        
    def get(self, row):
        # if row.Store != self.last_store:
        #     self.last_store = row.Store
        #     self.last = pd.to_datetime(np.nan)
        if (row[self.fld] == 1):
            self.last = row.Date
        return row.Date-self.last

class DurationTime(object):
    'iter across row'
    def __init__(self, fld):
        self.fld = fld
        self.dura = 0
        
    def get(self, row):
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
        df[prefix+fld] = df.apply(sh_el.get, axis=1).dt.days

        prefix = prefixs[-1]
        df = df.sort_values(['Date'], ascending=[False])
        init_date = df[df[fld] == 1].iloc[0]['Date']
        sh_el = TimeToEvent(fld, init_date)
        df[prefix+fld] = df.apply(sh_el.get, axis=1).dt.days
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