import pickle

import numpy as np
import word2vec


def train_word2vec(filename):
    word2vec.word2vec(filename, '{0}.bin'.format(filename), verbose=True)


with open('prob_table-yahoo-bus.json', 'r') as fin:
    import json

    db = json.load(fin)

model = word2vec.load('./yahoo-text.bin')

vocab = [unicode(i) for i in model.vocab]
vocab = set(vocab)

train_set = list()
missing_set = list()
for pair in db:
    if pair[0] in vocab and pair[1] in vocab:
        train_set.append((model[pair[0]], model[pair[1]], float(pair[2])))
    else:
        missing_set.append(tuple(pair))

train_x = list()
train_w = list()
train_y = list()
for line in train_set:
    train_x.append(line[0])
    train_y.append(line[1])
    train_w.append(line[2])
train_x = np.array(train_x)
train_y = np.array(train_y)
train_w = np.array(train_w)

with open('dataset-yahoo-bus.pkl', 'w') as fout:
    db = {'x': train_x, 'y': train_y, 'w': train_w}
    pickle.dump(db, fout)
