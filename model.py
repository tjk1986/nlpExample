from tensorflow.keras.layers import Dense, Embedding, LSTM, Concatenate, Dropout, Flatten, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.initializers import Constant
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model, Input

import os


def customModel(X_trainBody, X_trainTime, y_train, modelsDict, initial_bias):
    # first input
    i1 = Input(shape=(X_trainBody.shape[1],), name='subredditInput')
    x1 = Embedding(modelsDict['numberOfWords'], 9,
                   input_length=X_trainBody.shape[1])(i1)
    x1 = Conv1D(filters=32, kernel_size=3,
                padding='same', activation='relu')(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Bidirectional(LSTM(50, dropout=0.1, return_sequences=True))(x1)
    x1 = Bidirectional(LSTM(20, dropout=0.1, return_sequences=False))(x1)

    # second input
    i2 = Input(shape=(X_trainTime.shape[1],), name='timeInput')
    x2 = Dense(32, activation='relu')(i2)
    x2 = Dense(64, activation='relu')(x2)
    x2 = Dense(32, activation='relu')(x2)

    # concat layers
    concat_layer = Concatenate(name='timeAndText')([x1, x2])
    do = Dropout(0.5)(concat_layer)

    # outputs
    #output_1 = Dense(1, activation='sigmoid', name='score',
    #                 bias_initializer=Constant(initial_bias))(do)

    output_1 = Dense(1, activation='sigmoid', name='score')(do)

    output_2 = Dense(
        y_train[1].shape[1], activation='softmax', name='subreddit')(concat_layer)

    # model
    model = Model(inputs=[i1, i2], outputs=[output_1, output_2])

    # save model to png
    # cwd = os.getcwd()
    # filename = os.path.join(cwd, "model.png")
    # plot_model(model, to_file=filename)

    return model
