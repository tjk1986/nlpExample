# imports
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datetime import datetime
import numpy as np
import requests
import pickle
import json
import os

# local imports
from datasetBuilder import datasetBuilder
from textToVector import textToVector
from model import customModel
from trainModel import trainModel
from textParser import parser

# initialize dictionary
modelsDict = {}
modelsDict['numberOfWords'] = 20000
modelsDict['maxSequencesLen'] = 32

# settings
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
cwd = os.getcwd()

# read data
url = 'https://files.pushshift.io/reddit/comments/sample_data.json'
r = requests.get(url)
respString = r.content.decode('utf-8')
lines = respString.split("\n")
jsonData = [json.loads(line) for line in lines if line != '']

subredditList = ['politics', 'nfl', 'memes', 'AmItheAsshole', 'teenagers', 'CFB', 'dankmemes', 'modernwarfare', 'pokemontrades']

body = [i['body'] for i in jsonData if i['subreddit'] in subredditList]
time = [i['created_utc'] for i in jsonData if i['subreddit'] in subredditList]
score = [1 if i['score'] >= 10 else 0 for i in jsonData if i['subreddit'] in subredditList]
subreddit = [i['subreddit'] for i in jsonData if i['subreddit'] in subredditList]

# text preprocessing
body = list(map(parser, body))
body = [[lemmatizer.lemmatize(w) for w in l if w not in stop_words] for l in body]
body = [' '.join(filter(None, w)) for w in body]

# data to arrays
time = np.array([[datetime.fromtimestamp(epoch).hour,
                  datetime.fromtimestamp(epoch).minute] for epoch in time])
score = score = np.array(score)
subreddit = np.array(subreddit)
body = np.array(body, dtype=object)

# calculate initial bias
neg, pos = np.bincount(score)
initial_bias = np.log([pos / neg])
print(f'neg = {neg}, pos = {pos}')

# model inputs
X_trainBody, X_testBody, X_trainTime, X_testTime, y_train, y_test, modelsDict = datasetBuilder(
    body, score, subreddit, time, modelsDict)
del body, score, subreddit, time

# X to vectors
X_trainBody, X_testBody, modelsDict = textToVector(
    X_trainBody, X_testBody, modelsDict)

# Model
model = customModel(X_trainBody, X_trainTime,
                    y_train, modelsDict, initial_bias)

# save dictionary
cwd = os.getcwd()
filename = os.path.join(cwd, "modelsDict.sav")
with open(filename, 'wb') as handle:
    pickle.dump(modelsDict, handle)

# Train model
trainModel(model, X_trainBody, X_testBody,
           X_trainTime, X_testTime, y_train, y_test)
