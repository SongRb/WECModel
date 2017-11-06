# Runs on python3
from __future__ import print_function
import os
import pickle
import platform
import time
import random

import numpy as np
import tensorflow as tf


train_size = 0

# Aim to reduce cost, we use negative cosine function here
def cos_func(a, b):
    normalize_a = tf.nn.l2_normalize(a, 0)
    normalize_b = tf.nn.l2_normalize(b, 0)
    res = -tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
    return res


def trans(a):
    a = np.asarray(a)
    return a.transpose()


def load_model(filename):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(filename)
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        tf.initialize_all_variables().run()
        return sess.run('t_matrix:0')


def random_init():
    return np.random.uniform(low=-0.01, high=0.01, size=(DIMENSIONS, DIMENSIONS))


def saved_init(filename):
    return np.load(filename)

def find_model_id(model_dir):
    return sorted(next(os.walk(model_dir))[1],key=lambda x: int(''.join(x.split('-'))),reverse=True)

work_dir = os.path.join('work', 'L6-1')
dataset_name = 'WEC-L6-1-input.npz'
model_id = find_model_id(work_dir)

npzfile = np.load(os.path.join(work_dir, dataset_name))
train_x = npzfile['x']
train_y = npzfile['y']
train_w = npzfile['w']


print('Successfully load train dataset')
print('Processing data with tensorflow...')
# DIMENSIONS=100
DIMENSIONS = len(train_x[0])
POS_DS_SIZE = len(train_x)
# Creating training pair
# ds = [(train_x[i], train_y[i]) for i in range(DS_SIZE)]
# np.random.shuffle(ds)

SAMPLE_NUM=3
ds = list()
for i in xrange(POS_DS_SIZE):
    ds.append((train_x[i],train_y[i]))
    sample_count = 0
    while sample_count<SAMPLE_NUM:
        random_index = random.randint(0,POS_DS_SIZE-1)
        if random_index!=i:
            ds.append((train_x[i],-train_y[random_index]))
            sample_count+=1

np.random.shuffle(ds)

#SAMPLE_RATIO = 0.2
#ds=ds[0:int(SAMPLE_RATIO*len(ds))]


print('Positive and negative labels are generated...')
DS_SIZE = len(ds)
TRAIN_RATIO = 0.6  # 60% of the dataset is used for training
_train_size = int(DS_SIZE * TRAIN_RATIO)
_test_size = DS_SIZE - _train_size
STARTING_ALPHA = 0.25  # learning rate
ENDING_ALPHA = 0.001
ALPHA = 0.234
LAMBDA = 0.5  # L2 regularization factor
TRAINING_STEPS = 100001

DEBUG_STEPS = 200
ALPHA_STEP = STARTING_ALPHA/(TRAINING_STEPS/(2*DEBUG_STEPS))

train_data, train_labels = map(trans, zip(*ds[0:_train_size]))
train_size = len(train_data[0])
print('Train size: ',train_size)
test_data, test_labels = map(trans, zip(*ds[_train_size:]))
test_size = len(test_data[0])
print('Dataset generated')

# define the computational graph
print('Setting preferences')
graph = tf.Graph()
with graph.as_default():
    # declare graph inputs
    x_train = tf.placeholder(tf.float32, shape=(DIMENSIONS, train_size))
    y_train = tf.placeholder(tf.float32, shape=(DIMENSIONS, train_size))
    x_test = tf.placeholder(tf.float32, shape=(DIMENSIONS, test_size))
    y_test = tf.placeholder(tf.float32, shape=(DIMENSIONS, test_size))
    

    try:
        print(work_dir)
        theta = tf.Variable(
            np.load('t_matrix.npy'),
            dtype=np.float32, name='t_matrix')
    except IOError:
        print('Saved model not found, will train from start')
	theta = tf.Variable(
			np.random.uniform(low=-1, high=1, size=(DIMENSIONS, DIMENSIONS)),
    dtype=np.float32, name='t_matrix')


    # forward propagation
    train_prediction = tf.matmul(theta, x_train)
    test_prediction = tf.matmul(theta, x_test)
    train_cost = cos_func(train_prediction, y_train)
    optimizer = tf.train.GradientDescentOptimizer(ALPHA).minimize(train_cost)
    test_cost = cos_func(test_prediction, y_test)

# run the computation
print('Running...')
saver = tf.train.Saver({'t_matrix': theta})
with tf.Session(graph=graph) as s:
    tf.initialize_all_variables().run()
    print('Initialized')
    print(theta.eval())
    for step in range(3000,TRAINING_STEPS):
        _, train_c, test_c = s.run([optimizer, train_cost, test_cost],
                                   feed_dict={x_train: train_data,
                                              y_train: train_labels,
                                              x_test: test_data,
                                              y_test: test_labels})
        if step % DEBUG_STEPS == 0:
            print('\nAfter', step, 'iterations:')
            print('\tRelative train cost =',(train_size+train_c) / train_size)
            print('\tRelative test cost =', (test_size+test_c) / test_size)

            # Slightly decrease learning alpha
            ALPHA-=ALPHA_STEP
            if ALPHA<ENDING_ALPHA:ALPHA = ENDING_ALPHA
            print('\tLearning alpha changed into',ALPHA)

            # Save model
            time_str = time.strftime('%Y%m%d-%H%M%S')+'-'+str(step)
            os.makedirs(os.path.join(work_dir, time_str))
            save_path = saver.save(s, os.path.join(
                work_dir, time_str, 'model.ckpt'))
            with open(os.path.join(work_dir, time_str, 'status.txt'), 'w') as fout:
                fout.write('{0} {1} {2} {3}'.format(str(train_c),str(test_c),str(ALPHA),str(step)))
            np.save('t_matrix',s.run('t_matrix:0'))
            np.save(os.path.join(work_dir, time_str,
                                 't_matrix'), s.run('t_matrix:0'))
            print('Saved in', save_path)
