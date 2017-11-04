# Runs on python3
from __future__ import print_function
import os
import pickle
import platform
import time

import numpy as np
import tensorflow as tf


train_size = 0

# Aim to reduce cost, we use negative cosine function here
def cos_func(a, b):
    normalize_a = tf.nn.l2_normalize(a, 0)
    normalize_b = tf.nn.l2_normalize(b, 0)

    return tf.to_float(1, name='ToFloat') - tf.reduce_sum(tf.multiply(normalize_a, normalize_b))


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

work_dir = os.path.join('work', 'L6-1')
dataset_name = 'WEC-L6-1-input.npz'
model_id = '20171031-141035-13200'


npzfile = np.load(os.path.join(work_dir, dataset_name))
train_x = npzfile['x']
train_y = npzfile['y']
train_w = npzfile['w']


print('Successfully load train dataset')
print('Processing data with tensorflow...')
# DIMENSIONS=100
DIMENSIONS = len(train_x[0])
DS_SIZE = len(train_x)
# Creating training pair
ds = [(train_x[i], train_y[i]) for i in range(DS_SIZE)]
np.random.shuffle(ds)
TRAIN_RATIO = 0.6  # 60% of the dataset is used for training
_train_size = int(DS_SIZE * TRAIN_RATIO)
_test_size = DS_SIZE - _train_size
ALPHA = 1e-5  # learning rate
LAMBDA = 0.5  # L2 regularization factor
TRAINING_STEPS = 20001
# Swap train data and test data
train_data, train_labels = map(trans, zip(*ds[0:_train_size]))
train_size = len(train_data[0])
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
    
    # theta = tf.Variable(
    # np.random.uniform(low=-0.1, high=0.1, size=(DIMENSIONS, DIMENSIONS)),
    # dtype=np.float32, name='t_matrix')

    theta = tf.Variable(
        saved_init(os.path.join(
            work_dir, model_id, 't_matrix.npy')),
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
    print("Initialized")
    print(theta.eval())
    for step in range(TRAINING_STEPS):
        _, train_c, test_c = s.run([optimizer, train_cost, test_cost],
                                   feed_dict={x_train: train_data,
                                              y_train: train_labels,
                                              x_test: test_data,
                                              y_test: test_labels})
        if step % 200 == 0:
            print("\nAfter", step, "iterations:")
            print("  Relative train cost =", train_c / train_size)
            print("  Relative test cost =", test_c / test_size)
            time_str = time.strftime("%Y%m%d-%H%M%S")+'-'+str(step)
            os.makedirs(os.path.join(work_dir, time_str))
            save_path = saver.save(s, os.path.join(
                work_dir, time_str, 'model.ckpt'))
            with open(os.path.join(work_dir, time_str, 'status.txt'), 'w') as fout:
                fout.write('{0} {1}'.format(str(train_c),str(test_c)))
            np.save(os.path.join(work_dir, time_str,
                                 't_matrix'), s.run('t_matrix:0'))
            print("Saved in", save_path)
