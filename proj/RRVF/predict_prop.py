import pandas as pd
import numpy as np
from fbprophet import Prophet

import os
# os.chdir(r'E:\workspace\ai\ml_utils\proj\RRVF')
os.chdir(r'/Users/kevindu/Documents/workspace/ml_utils/proj/RRVF')
import glob
import re
import pickle

import numpy as np
import pandas as pd
from isoweek import Week
from pandas_summary import DataFrameSummary
from keras.models import model_from_yaml
import utils
# import xgboost
import random


data_dir = r'./data'
tst_time = pd.read_csv('time.csv')
model_zoo = pickle.load(open('result/prop.pkl', 'rb'))
result = {}
for model_key in model_zoo.keys():
    model = model_zoo[model_key]
    res = np.exp(model.predict(tst_time)['yhat'])
    print('Done {} sample {}'.format(model_key, res[0]))
    result[model_key] = res
import pickle as pkl
pkl.dump(result, open('result/result.pkl','wb'))
