from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np


def datasetBuilder(body, score, subreddit, time, modelsDict):

    subreddit = subreddit.reshape(-1, 1)

    # subreddit encoder
    encSubreddit = OneHotEncoder(
        handle_unknown='ignore', sparse=False, dtype=np.int8)
    encSubreddit.fit(subreddit)
    subreddit = encSubreddit.transform(subreddit)

    # time encoder
    encTime = OneHotEncoder(handle_unknown='ignore',
                            sparse=False, dtype=np.int8)
    encTime.fit(time)
    time = encTime.transform(time)

    # test size
    testSize = int(len(body) * 0.7)

    # inputs
    X_trainBody = body[:testSize]
    X_testBody = body[testSize:]
    X_trainTime = time[:testSize]
    X_testTime = time[testSize:]

    # outputs
    y_train = [score[:testSize], subreddit[:testSize]]
    y_test = [score[testSize:], subreddit[testSize:]]

    modelsDict['encSubreddit'] = encSubreddit
    modelsDict['encTime'] = encTime

    return X_trainBody, X_testBody, X_trainTime, X_testTime, y_train, y_test, modelsDict
