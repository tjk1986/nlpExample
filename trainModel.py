from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, CategoricalAccuracy, Precision, Recall, AUC, BinaryAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle
import json
import os


def trainModel(model, X_trainBody, X_testBody, X_trainTime, X_testTime, y_train, y_test):

    metricsScore = [TruePositives(name='tp'),
                    FalsePositives(name='fp'),
                    TrueNegatives(name='tn'),
                    FalseNegatives(name='fn'),
                    BinaryAccuracy(name='accuracy'),
                    Precision(name='precision'),
                    Recall(name='recall'),
                    AUC(name='auc')]

    metricsSubreddit = [TruePositives(name='tp'),
                        FalsePositives(name='fp'),
                        TrueNegatives(name='tn'),
                        FalseNegatives(name='fn'),
                        CategoricalAccuracy(name='categorical_accuracy'),
                        Precision(name='precision'),
                        Recall(name='recall'),
                        AUC(name='auc')]

    metrics_array = {'score': metricsScore, 'subreddit': metricsSubreddit}
    loss_array = {'score': 'binary_crossentropy',
                  'subreddit': 'categorical_crossentropy'}

    cwd = os.getcwd()
    filepath = os.path.join(cwd, "models/saved-model-{epoch:02d}.hdf5")

    checkpoint = ModelCheckpoint(filepath, verbose=1,
                                 mode='auto', save_freq='epoch')

    earlystop_callback = EarlyStopping(
        monitor='val_subreddit_categorical_accuracy', min_delta=0.0001, patience=2)

    model.compile(optimizer=Adam(0.001),
                  loss=loss_array,
                  metrics=metrics_array)

    sampleWeight = {'score': {}}

    scoreWeight = compute_class_weight('balanced', np.unique(y_train[0]), y_train[0])
    sample_weight = np.ones(shape=(len(y_train[0]),))
    sample_weight[y_train[0] == 0] = scoreWeight[0]
    sample_weight[y_train[0] == 1] = scoreWeight[1]

    sampleWeight['score'] = sample_weight
    print(scoreWeight)

    history = model.fit([X_trainBody, X_trainTime], y_train, epochs=20, validation_data=([X_testBody, X_testTime], y_test),
                        batch_size=2, sample_weight=sampleWeight, callbacks=[checkpoint, earlystop_callback])

    with open('history.sav', 'wb') as outfile:
        pickle.dump(history.history, outfile)
