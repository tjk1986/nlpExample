from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def textToVector(X_trainBody, X_testBody, modelsDict):
    # tokenizer
    tokenizer = Tokenizer(num_words=modelsDict['numberOfWords'], oov_token='<OOV>')
    tokenizer.fit_on_texts(X_trainBody)

    # train
    X_trainBody = tokenizer.texts_to_sequences(X_trainBody)
    X_trainBody = pad_sequences(X_trainBody, padding='post', maxlen=modelsDict['maxSequencesLen'])

    # test
    X_testBody = tokenizer.texts_to_sequences(X_testBody)
    X_testBody = pad_sequences(X_testBody, padding='post', maxlen=modelsDict['maxSequencesLen'])

    # tokenizer to dictionary
    modelsDict['tokenizer'] = tokenizer

    return X_trainBody, X_testBody, modelsDict
