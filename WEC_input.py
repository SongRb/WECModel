import pickle

import numpy as np
import word2vec
import os
import json

from model_settings import *

def train_word2vec(filename):
    word2vec.word2vec(filename, '{0}.bin'.format(filename), verbose=True)



with open(os.path.join(DATA_PATH,'prob_table-yahoo-bus.json'), 'r') as fin:
    db = json.load(fin)

model = word2vec.load(os.path.join(DATA_PATH,'yahoo-text.bin'))

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

with open(os.path.join(DATA_PATH,'dataset-yahoo-bus.pkl'), 'w') as fout:
    db = {'x': train_x, 'y': train_y, 'w': train_w}
    pickle.dump(db, fout)
