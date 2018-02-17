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
import pandas as pd
import numpy as np
import utils

PATH = 'data/'  # "../../../data/RRVF/"  # 'data/'
RESULT = "result/" # "../../../data/RRVF/" # "result/"


def load_splits():
    "load_splits"
    splits = pickle.load(open(f'{RESULT}_datasplits.pkl', 'rb'))
    contin_vars = splits['contin_vars']
    cat_vars = splits['cat_vars']
    X_train = splits['trn']
    y_train = splits['trn_y']
    X_valid = splits['val']
    y_valid = splits['val_y']
    X_test = splits['test']
    return X_train, y_train, X_valid, y_valid, X_test, cat_vars, contin_vars

def generate_sub(csv_fn, m, df_test):
    pred_test= m.predict(df_test)
    pred_test = np.exp(pred_test)
    test_set = df_test.copy()
    test_set['visitors']=pred_test
    trn_like_test = test_set.reset_index()[['air_store_id', 'visit_date', 'visitors']]
    trn_like_test.visit_date = trn_like_test.visit_date.astype('str')
    sub = utils.trn2test(trn_like_test)
    sub.to_csv(csv_fn, index=False)

def rmsle(x, y):
    # np.log(targ + 1) - np.log(y_pred + 1)
    return math.sqrt(((x-y)**2).mean())


def rmsle_wo_log_sk(y_predicted, y_true):
    y_predicted_orig = utils.log_max_inv(y_predicted, max_log_y)
    y_true_orig = utils.log_max_inv(y_true, max_log_y)
    score = rmsle(y_predicted_orig, y_true_orig)
    return score

def rmsle_wo_log(y_predicted, y_true):
    y_true = y_true.get_label()
    score = rmsle(y_predicted, y_true)
    return ('rmsle', score, False)


if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, X_test, cat_vars, contin_vars = load_splits()
    print('Trn size {}, Val size {}'.format(len(X_valid), len(y_valid)))
    print('Trn store # {}, Val store # {}'.format(
        len(X_train.air_store_id.unique()), len(X_valid.air_store_id.unique())))
    lgb_train = lgb.Dataset(X_train, y_train.ravel(), free_raw_data=False)
    lgb_eval = lgb.Dataset(X_valid, y_valid.ravel(),
                        reference=lgb_train, free_raw_data=False)

    # specify your configurations as a dict
    params = {
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        # 'metric': 'rmse',
        'sub_feature': 0.7,
        'num_leaves': 60,
        'min_data': 100,
        'min_hessian': 1,
        # 'verbose': -1,
    }
    evals_result = {}
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=500,
                    valid_sets=(lgb_train, lgb_eval),
                    feval=rmsle_wo_log,
                    evals_result=evals_result,
                    )  # early_stopping_rounds=0

    gbm.save_model('./result/gbm_model.txt')
    csv_fn = 'lgb.csv'
    generate_sub(csv_fn, gbm, X_test)
    print('Done')
#     y_train_orig = train_set.visitors.values

#     base_valid = rmsle(g_y_trn.values, y_train_orig.ravel())
#     base_trn = rmsle(g_y_valid.values, y_valid_orig.ravel())

#     print('Base line train loss {}, valid loss {}'.format(base_trn, base_valid))

# pred_valid = gbm.predict(X_valid)
# pred_valid_orig = utils.log_max_inv(pred_valid, max_log_y)
# valid_loss = rmsle(pred_valid_orig, y_valid_orig.ravel())
# pred_trn = gbm.predict(X_train)
# pred_trn_orig = utils.log_max_inv(pred_trn, max_log_y)
# trn_loss = rmsle(pred_trn_orig, y_train_orig.ravel())
# print('LightBGM train loss: {}, valid loss: {}'.format(trn_loss, valid_loss))


# data_dir = r'./data'
# # test = pd.read_csv('{}/sample_submission.csv'.format(data_dir))
# # trn_like_test = utils.tes2trn(test)
# # trn_like_test = trn_like_test.assign(visitors = np.round(np.random.rand(len(trn_like_test)) * 100))
# test = pd.read_csv('{}/air_visit_data.csv'.format(data_dir))
# test = test[test.air_store_id.isin(test_stores)]
# test_data = utils.get_data(
#     test, contin_map_fit=contin_map_fit, cat_map_fit=cat_map_fit)
# X_tst = test_data['X']
# y_tst = test_data['Y']
# g_y_tst = test_data['gadge']

# pred_gbm = utils.log_max_inv(gbm.predict(X_tst), max_log_y)
# y_tst_orig = test.visitors.values
# tst_loss = rmsle(pred_gbm, y_tst_orig.ravel())
# print("Train loss corrected loss {}".format(tst_loss))
