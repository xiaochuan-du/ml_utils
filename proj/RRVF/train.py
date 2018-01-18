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

if __name__ == '__main__':
    model_file, usewgt = 'newmodel_bn', True
    data_dir = r'./data'
    trn = pd.read_csv('{}/air_visit_data.csv'.format(data_dir))
    # use_cacheing
    feas = utils.data2fea(
        trn, data_dir, run_para={"af_etl": 'result/af_etl.csv'})
    nn_fea = feas['nn_fea']

    y = feas['y']
    contin_cols = feas['contin_cols']
    cat_map_fit = feas['cat_map_fit']
    tidy_data = feas['tidy_data']
    date_sr = pd.to_datetime(tidy_data.Date)
    dat = utils.data_split_by_date(
        nn_fea, y, date_sr, trn2val_ratio=9, step_days=50)
    # valid & trn splitting
    dat_d = dat[0]
    map_train, y_train_orig, map_valid, y_valid_orig = dat_d['x_trn'], dat_d['y_trn'], dat_d['x_valid'], dat_d['y_valid']
    y_train, y_valid, max_log_y = utils.uniform_y(y_train_orig, y_valid_orig)

    model = utils.get_bn_model(contin_cols, cat_map_fit)
    yaml_string = model.to_yaml()
    with open('./result/{}.yml'.format(model_file), 'w') as file_obj:
        file_obj.write(yaml_string)
    if usewgt:
        model.load_weights('./result/{}.h5'.format(model_file))
    seed = np.random.rand() * (1) - 5
    model.optimizer.lr, epochs = np.power(10, seed), 500  # np.power(10, seed)
    tensorboard = TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        write_graph=True,
        write_images=False)
    checkpoint = ModelCheckpoint(
        './result/{}.h5'.format(model_file),
        save_best_only=True,
        monitor='loss',
        save_weights_only=True)
    hist = model.fit(
        map_train,
        y_train,
        batch_size=128,
        epochs=epochs,
        validation_data=(map_valid, y_valid),
        callbacks=[tensorboard, checkpoint])

    # model.save_weights('./result/caching.h5')
    model.evaluate(map_valid, y_valid)
    # pickle.dump(hist, open('./result/hist1.pkl', 'wb'))
