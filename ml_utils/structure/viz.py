import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pandas_summary import DataFrameSummary

#%%
#Visualize seasonality of given time series data
def plot_seasonality(df, ds_col='ds', y_col='y', figsize=(20, 10)):
  # reformat data
  data = df[[ds_col, y_col]].rename({
    ds_col: 'ds',
    y_col: 'y'
  }, axis='columns')
  
  # force ds to be datetime
  sm_df= DataFrameSummary(data).summary()
  if sm_df.loc['types'].ds != 'date':
    data['ds'] = pd.to_datetime(data.ds)

  #add DoW info
  data['dow'] = data.ds.dt.dayofweek
  dow_label = {
    '0': 'Mon',
    '1': 'Tues',
    '2': 'Wed',
    '3': 'Thurs',
    '4': 'Fri',
    '5': 'Sat',
    '6': 'Sun'
  }
  data['dow_label'] = data.dow.apply(lambda x : dow_label[str(x)])
  
  #plot
  plt.figure(figsize=figsize)
  subplots = []
  for i in range(7):
    data_sub = data[data.dow == i]
    subplots.append(plt.plot(data_sub.ds, data_sub.y, 'o')[0])
  plt.figlegend(tuple(subplots), tuple(dow_label.values()), 'center right')
