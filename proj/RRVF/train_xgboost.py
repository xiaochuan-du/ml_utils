import glob
import re
import pickle

import numpy as np
import pandas as pd
from isoweek import Week
from pandas_summary import DataFrameSummary
from keras.models import model_from_yaml
import utils
from keras.callbacks import TensorBoard, ModelCheckpoint
import xgboost
import random

if __name__ == '__main__':
    data_dir = r'./data'
    trn = pd.read_csv('{}/air_visit_data.csv'.format(data_dir))
    feas = utils.data2fea(trn, data_dir)
    input_map = feas['x_map']
    y = feas['y']
    contin_cols = feas['contin_cols']
    cat_map_fit = feas['cat_map_fit']
    # valid & trn splitting
    x_fit = feas['x_fit']
    all_vars = feas['all_vars']
    xgb_parms = {'learning_rate': 0.1, 'subsample': 0.6, 
             'colsample_bylevel': 0.6, 'silent': True, 'objective': 'reg:linear'}
    X_train, y_train_orig = x_fit[:20000], y[:20000]
    X_valid, y_valid_orig = x_fit[20000:], y[20000:] 
    y_train, y_valid, max_log_y = utils.uniform_y(y_train_orig, y_valid_orig)

    xdata = xgboost.DMatrix(X_train, y_train, feature_names=all_vars)
    xdata_val = xgboost.DMatrix(X_valid, y_valid, feature_names=all_vars)
    xgb_parms['seed'] = random.randint(0,1e9)
    model = xgboost.train(xgb_parms, xdata)
    print(model.eval(xdata_val))
    # model.save_weights('./result/caching.h5')
    # model.evaluate(map_valid, y_valid)
    # pickle.dump(hist, open('./result/hist1.pkl', 'wb'))
