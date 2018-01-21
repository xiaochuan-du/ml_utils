import pandas as pd
import numpy as np
from fbprophet import Prophet
import glob
import re
import pickle
from multiprocessing import Pool
import numpy as np
import pandas as pd
from isoweek import Week
from pandas_summary import DataFrameSummary
from keras.models import model_from_yaml
import utils
# import xgboost
import random

concurrency = 3
data_dir = r'./data'
tst_time = pd.read_csv('time.csv').rename(
    {
        'visit_date': 'ds',
    }, axis="columns"
)
step_task = 10
model_zoo = pickle.load(open('result/prop.pkl', 'rb'))
result = {}
model_keys = list(model_zoo.keys())

def func(model_key):
    model = model_zoo[model_key]
    res = np.exp(model.predict(tst_time)['yhat'])
    return res

num_tasks = len(model_keys)

with Pool(concurrency) as pool:
    for i, res in enumerate(pool.imap(func, model_keys), 1):
        result[model_keys[i-1]] = res
        if i % step_task == 0:
            print("progress={} %".format(i/num_tasks*100))


import pickle as pkl
print('dump result')
pkl.dump(result, open('result/result1.pkl','wb'))
