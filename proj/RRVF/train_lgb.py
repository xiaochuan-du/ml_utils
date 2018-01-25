import glob
import re
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from isoweek import Week
from pandas_summary import DataFrameSummary
import itertools
import utils
import lightgbm as lgb
import random
# from keras.models import model_from_yaml
from pandas_summary import DataFrameSummary

import math
def rmsle(y_pred, targ):
    log_vars = np.log(targ + 1) - np.log(y_pred + 1)
    return math.sqrt(np.square(log_vars).mean())
def plot_impt(model):
    importance = model.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance');
    # df[df.fscore < 0.009].feature.tolist()

def rmsle_wo_log(y_predicted, y_true):
    y_true = y_true.get_label()
    y_predicted_orig = utils.log_max_inv(y_predicted, max_log_y)
    y_true_orig = utils.log_max_inv(y_true, max_log_y)
    score = rmsle(y_predicted_orig, y_true_orig)
    return ('rmsle', score, False)

def rmsle_wo_log_sk(y_predicted, y_true):
    y_predicted_orig = utils.log_max_inv(y_predicted, max_log_y)
    y_true_orig = utils.log_max_inv(y_true, max_log_y)
    score = rmsle(y_predicted_orig, y_true_orig)
    return score


def split(df):
    trn_len = int(np.floor(len(df) * 0.9))
    valid_len = len(df) - trn_len
    df['type'] = 0  #0 for train 1 for valid
    indexs = df.index
    df = df.reset_index()
    df.loc[trn_len:, 'type'] =  1
    return df
data_raw = pd.read_csv('./data/air_visit_data.csv')
test = pd.read_csv('./data/sample_submission.csv')
test_data = utils.tes2trn(test)
test_stores = test_data.air_store_id.unique()
data = data_raw[data_raw.air_store_id.isin(test_stores)]
tag_data = data.groupby('air_store_id').apply(split)
t = tag_data.set_index('index')
train_set = t[t.type == 0]
valid_set = t[t.type == 1]
train_set = train_set.reset_index().drop(['index', 'type'], axis=1)
valid_set = valid_set.reset_index().drop(['index', 'type'], axis=1)

print('Trn size {}, Val size {}'.format(len(train_set), len(valid_set)))
print('Trn store # {}, Val store # {}'.format(len(train_set.air_store_id.unique()), len(valid_set.air_store_id.unique())))

trn_data = utils.get_data(train_set)
X_train = trn_data['X']
y_train = trn_data['Y']
y_train_orig = trn_data['Y_org']
max_log_y = trn_data['max_log_y']
g_y_trn = trn_data['gadge']
contin_map_fit = trn_data['contin_map_fit']
cat_map_fit = trn_data['cat_map_fit']
all_vars = trn_data['all_vars']


val_data = utils.get_data(valid_set, contin_map_fit=contin_map_fit, cat_map_fit=cat_map_fit)
X_valid = val_data['X']
y_valid = val_data['Y']
y_valid_orig = val_data['Y_org']
g_y_valid = val_data['gadge']


lgb_train = lgb.Dataset(X_train, y_train.ravel(), free_raw_data=False)
lgb_eval = lgb.Dataset(X_valid, y_valid.ravel(), reference=lgb_train, free_raw_data=False)

# specify your configurations as a dict
params = {
    'task': 'train',
#     'boosting_type': 'dart',
    'objective': 'regression',
#     'metric': {'mse'},
    'num_leaves': 80,
    'learning_rate': 0.08,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'max_bin': 15,
    'max_depth': 40
}
evals_result = {} 
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=300,
                valid_sets=(lgb_train, lgb_eval),
                feval=rmsle_wo_log,
                evals_result=evals_result,
                ) # early_stopping_rounds=0

y_train_orig = train_set.visitors.values

base_valid= rmsle(g_y_trn.values, y_train_orig.ravel())
base_trn= rmsle(g_y_valid.values, y_valid_orig.ravel())

print('Base line train loss {}, valid loss {}'.format(base_trn, base_valid))

# split_point = X.shape[0] - y_valid_orig.ravel().shape[0]
# base_line = rmsle(tidy_data['prop_yhat'].values[split_point:].ravel(), y_valid_orig.ravel())

pred_valid = gbm.predict(X_valid)
pred_valid_orig = utils.log_max_inv(pred_valid, max_log_y)
valid_loss = rmsle(pred_valid_orig, y_valid_orig.ravel())
pred_trn = gbm.predict(X_train)
pred_trn_orig = utils.log_max_inv(pred_trn, max_log_y)
trn_loss = rmsle(pred_trn_orig, y_train_orig.ravel())
print('LightBGM train loss: {}, valid loss: {}'.format(trn_loss, valid_loss))


data_dir = r'./data'
# test = pd.read_csv('{}/sample_submission.csv'.format(data_dir))
# trn_like_test = utils.tes2trn(test)
# trn_like_test = trn_like_test.assign(visitors = np.round(np.random.rand(len(trn_like_test)) * 100))
test = pd.read_csv('{}/air_visit_data.csv'.format(data_dir))
test = test[test.air_store_id.isin(test_stores)]
test_data = utils.get_data(test, contin_map_fit=contin_map_fit, cat_map_fit=cat_map_fit)
X_tst= test_data['X']
y_tst = test_data['Y']
g_y_tst = test_data['gadge']

pred_gbm = utils.log_max_inv(gbm.predict(X_tst), max_log_y)
y_tst_orig = test.visitors.values
tst_loss = rmsle(pred_gbm, y_tst_orig.ravel())
print("Train loss corrected loss {}".format(tst_loss))