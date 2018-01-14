import glob
import re

import numpy as np
import pandas as pd
from isoweek import Week
from pandas_summary import DataFrameSummary
import utils



if __name__ == '__main__':
    data_dir = r'/Users/kevindu/Documents/workspace/ml_utils/proj/RRVF/data'
    trn = pd.read_csv('{}/air_visit_data.csv'.format(data_dir))
    feas = utils.data2fea(trn, data_dir)
    input_map = feas['x_map']
    y = feas['y']
    contin_cols = feas['contin_cols']
    cat_map_fit = feas['cat_map_fit']
    # valid & trn splitting
    map_train, map_valid, y_train_orig, y_valid_orig = utils.ts_data_split(input_map, y)
    y_train, y_valid, max_log_y = utils.uniform_y(y_train_orig, y_valid_orig)

    model = utils.get_model(contin_cols, cat_map_fit)
    model.optimizer.lr = 1e-2
    hist = model.fit(
        map_train,
        y_train,
        batch_size=128,
        epochs=1,
        validation_data=(map_valid, y_valid))

    hist.model.save_weights('./result/caching.h5')
    model.evaluate(map_valid, y_valid)
