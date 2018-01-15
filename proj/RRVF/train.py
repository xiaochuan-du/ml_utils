import glob
import re
import pickle

import numpy as np
import pandas as pd
from isoweek import Week
from pandas_summary import DataFrameSummary
from keras.models import model_from_yaml
import utils



if __name__ == '__main__':
    data_dir = r'/src/data'
    trn = pd.read_csv('{}/air_visit_data.csv'.format(data_dir))
    feas = utils.data2fea(trn, data_dir)
    input_map = feas['x_map']
    y = feas['y']
    contin_cols = feas['contin_cols']
    cat_map_fit = feas['cat_map_fit']
    ts_date = feas['times']
    s_i = ts_date[ts_date == '2016-04-23'].index[0]
    e_i = ts_date[ts_date == '2016-06-01'].index[0]
    # valid & trn splitting
    map_train, map_valid, y_train_orig, y_valid_orig = utils.ts_data_split(input_map, y, s_i, e_i)
    y_train, y_valid, max_log_y = utils.uniform_y(y_train_orig, y_valid_orig)

    model = utils.get_model(contin_cols, cat_map_fit)
    yaml_string = model.to_yaml()
    with open('./result/model.yml', 'w') as file_obj:
        file_obj.write(yaml_string)
    model.optimizer.lr = 1e-2
    hist = model.fit(
        map_train,
        y_train,
        batch_size=128,
        epochs=6,
        validation_data=(map_valid, y_valid))

    model.save_weights('./result/caching.h5')
    model.evaluate(map_valid, y_valid)
    # pickle.dump(hist, open('./result/hist1.pkl', 'wb'))
