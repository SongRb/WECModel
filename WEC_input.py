# python2
import pickle

import numpy as np
import word2vec
import os
import json

from model_settings import *

def train_word2vec(filename):
    return word2vec.word2vec(filename, '{0}.bin'.format(filename), verbose=True)

filename = 'work/L6-1/YahooTextL6-1'
train_word2vec(filename)

with open(os.path.join(DATA_PATH,'prob_table-L6-1-a2q.json'), 'r') as fin:
    db = json.load(fin)

model = word2vec.load('{0}.bin'.format(filename))
# model = train_word2vec(os.path.join(DATA_PATH,'yahoo-text-full'))
vocab = [unicode(i) for i in model.vocab]
vocab = set(vocab)

train_set = list()
missing_set = list()
for pair in db:
    if pair[0] in vocab and pair[1] in vocab:
        train_set.append((model[pair[0]], model[pair[1]], float(pair[2])))
    else:
        missing_set.append(tuple(pair))


train_x,train_y,train_w = map(np.array,zip(*train_set))

np.savez('WEC-L6-1-input',x=train_x,y=train_y,w=train_w)
