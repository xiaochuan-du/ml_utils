#%%
import matplotlib.pyplot as plt
import matplotlib as plt
import pandas as pd
import numpy as np
# matplotlib inline
from IPython.display import HTML
from isoweek import Week
from pandas_summary import DataFrameSummary

#%%
import os
os.chdir('/Users/kevindu/Documents/workspace/ml_utils/proj/RRVF')

#%%
air_reserve = pd.read_csv('./data/air_reserve.csv')
air_store_info = pd.read_csv('./data/air_store_info.csv')
air_visit_data = pd.read_csv('./data/air_visit_data.csv')
date_info = pd.read_csv('./data/date_info.csv')
hpg_reserve = pd.read_csv('./data/hpg_reserve.csv')
hpg_store_info = pd.read_csv('./data/hpg_store_info.csv')
store_id_relation = pd.read_csv('./data/store_id_relation.csv')
print('done')

#%%
air_visit_data.visit_date = pd.to_datetime(air_visit_data.visit_date)
air_visit_data = air_visit_data.set_index('visit_date')

#%%
tbls = {"air_reserve": air_reserve, "air_store_info": air_store_info,
        "air_visit_data": air_visit_data, "date_info": date_info, "hpg_reserve": hpg_reserve,
        "hpg_reserve": hpg_reserve, "hpg_store_info": hpg_store_info, "store_id_relation": store_id_relation}
for tbl, df in tbls.items():
    print(tbl)
    display(df.head())

#%%
# joined = pd.merge(air_store_info, store_id_relation, how='left', )
# hpg_store_info = hpg_store_info.rename({'latitude': 'hpb_latitude', 'longitude': 'hpb_longitude'}, axis='columns')
# joined = pd.merge(joined, hpg_store_info, how='left', suffixes=('air_', 'hpg_'))
hpg_joined = pd.merge(hpg_reserve, hpg_store_info, how='left', )
display(hpg_joined.head())

#%%
air_joined = pd.merge(air_reserve, air_store_info, how='left', )
display(air_joined.head())

#%%
hpg_fl_joined = pd.merge(store_id_relation, hpg_joined, how='left', )
display(hpg_fl_joined.head())

#%%
hpg_fl_joined.rename(
    {
        'hpb_latitude': 'latitude', 
        'hpb_longitude': 'longitude',
        'hpg_genre_name': 'genre_name',
        'hpg_area_name': 'area_name',
}, axis='columns', inplace=True)
hpg_fl_joined.drop('hpg_store_id',axis=1, inplace=True, errors="ignore")
hpg_fl_joined = hpg_fl_joined.assign(src='hpg')

display(hpg_fl_joined.head())

#%%
air_fl_joined = air_joined
air_fl_joined.rename(
    {
        'air_genre_name': 'genre_name',
        'air_area_name': 'area_name',
}, axis='columns', inplace=True)
air_fl_joined = air_fl_joined.assign(src='air')
display(air_fl_joined.head())

#%%
reserve = pd.concat([air_fl_joined, hpg_fl_joined], axis=0)
display(reserve.head())

#%%
reserve.visit_datetime = pd.to_datetime(reserve.visit_datetime )
reserve.reserve_datetime = pd.to_datetime(reserve.reserve_datetime )
reserve.to_pickle('./result/reserve_row.pkl')

#%%
reserve = pd.read_pickle('./result/reserve_row.pkl')
display(reserve.head())




#%%
# date_info.calendar_date = pd.to_datetime(date_info.calendar_date)
date_info.drop('day_of_week',axis=1, inplace=True, errors="ignore")
date_info.calendar_date = date_info.calendar_date.astype('str')
display(date_info.head())

#%%
reserve_en = None
reserve_en = reserve.assign(visit_date=reserve.visit_datetime.dt.date)
reserve_en.visit_date = reserve_en.visit_date.astype('str')
reserve_en = pd.merge(reserve_en, date_info, how='left',
    left_on=['visit_date'], right_on=['calendar_date'])
reserve_en.rename(
    {
        'holiday_flg': 'visit_holiday_flg', 
}, axis='columns', inplace=True)
reserve_en.drop('calendar_date',axis=1, inplace=True, errors="ignore")

reserve_en = reserve_en.assign(reserve_date=reserve_en.reserve_datetime.dt.date)
reserve_en.reserve_date = reserve_en.reserve_date.astype('str')
reserve_en = pd.merge(reserve_en, date_info, how='left',
    left_on=['reserve_date'], right_on=['calendar_date'])
reserve_en.rename(
    {
        'holiday_flg': 'reserve_holiday_flg', 
}, axis='columns', inplace=True)
reserve_en.drop('calendar_date',axis=1, inplace=True, errors="ignore")
display(reserve_en.head())

#%%
reserve_en.to_pickle('./result/reserve_en.pkl')

#%%
reserve_en = pd.read_pickle('./result/reserve_en.pkl')
air_visit_data = pd.read_csv('./data/air_visit_data.csv')
date_info = pd.read_csv('./data/date_info.csv')

#%%
tbls = {
        "air_visit_data": air_visit_data, "date_info": date_info, 
        "reserve_en": reserve_en}
for tbl, df in tbls.items():
    print(tbl)
    display(df.head())

#%%
air_visit_data[air_visit_data.air_store_id == 'air_ba937bf13d40fb24'].visitors.plot()

#%%
ot_trn = pd.read_pickle('./result/ot_trn.pkl')
display(ot_trn.head())

#%%
ot_trn.air_area_name2.hist()