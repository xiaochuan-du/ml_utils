"""
preprocessing steps for general learning problems
"""
from keras.layers import (Dense, Flatten, MaxPooling1D, Dropout, Conv1D,
                          Embedding, Input, Concatenate)
from keras.models import Sequential, Model
from keras.optimizers import Adam


def get_cnn_txt_model(emb, emb_len, vocab_size, seq_len):
    """
    in order to let emb fit into target corpus, it is necessary to train
    emb after prev training
    model.fit(trn, labels_train, validation_data=(test, labels_test),
        nb_epoch=2, batch_size=64)
    model.layers[0].trainable=True
    model.optimizer.lr=1e-4
    model.fit(trn, labels_train, validation_data=(test, labels_test),
        nb_epoch=1, batch_size=64)
    """
    conv = Sequential([
        Embedding(
            vocab_size,
            emb_len,
            input_length=seq_len,
            dropout=0.2,
            weights=[emb],
            trainable=False), Dropout(0.2), Conv1D(
                64, 5, border_mode='same',
                activation='relu'), Dropout(0.2), MaxPooling1D(), Flatten(),
        Dense(100, activation='relu'), Dropout(0.7), Dense(
            1, activation='sigmoid')
    ])
    conv.compile(
        loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return conv


def get_muld_graph_cnn_txt_model(emb,
                                 emb_len,
                                 vocab_size,
                                 seq_len,
                                 dran=(3, 6)):
    """ get multi dimension cnn for text
    model.fit(trn, labels_train, validation_data=(test, labels_test),
        epochs=2, batch_size=64)
    model.layers[0].trainable=False
    model.optimizer.lr=1e-5
    model.fit(trn, labels_train, validation_data=(test, labels_test),
        epochs=2, batch_size=64)
    """
    graph_in = Input((vocab_size, emb_len))
    convs = []
    for fsz in range(dran):
        x = Conv1D(64, fsz, border_mode='same', activation="relu")(graph_in)
        x = MaxPooling1D()(x)
        x = Flatten()(x)
        convs.append(x)
    out = Concatenate()(convs)
    graph = Model(graph_in, out)
    model = Sequential([
        Embedding(
            vocab_size,
            emb_len,
            input_length=seq_len,
            dropout=0.2,
            weights=[emb]), Dropout(0.2), graph, Dropout(0.5), Dense(
                100, activation="relu"), Dropout(0.7), Dense(
                    1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model
