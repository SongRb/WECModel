# Runs on python3
import os
import pickle
import platform
import time

import numpy as np
import tensorflow as tf


def cos_func(a, b):
    normalize_a = tf.nn.l2_normalize(a, 0)
    normalize_b = tf.nn.l2_normalize(b, 0)
    return -tf.reduce_sum(tf.multiply(normalize_a, normalize_b))

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


npzfile = np.load(os.path.join(work_dir,'WEC-L6-1-input.npz'))
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
ALPHA = 3e-7  # learning rate
LAMBDA = 0.5  # L2 regularization factor
TRAINING_STEPS = 20001
# Swap train data and test data
train_data, train_labels = map(trans,zip(*ds[_train_size:]))

test_data, test_labels =map(trans, zip(*ds[0:_train_size]))

print('Dataset generated')

# define the computational graph
print('Setting preferences')
graph = tf.Graph()
with graph.as_default():
    # declare graph inputs
    # x_train = tf.placeholder(tf.float32, shape=(_train_size, DIMENSIONS))
    # y_train = tf.placeholder(tf.float32, shape=(_train_size, DIMENSIONS))
    # x_test = tf.placeholder(tf.float32, shape=(_test_size, DIMENSIONS))
    # y_test = tf.placeholder(tf.float32, shape=(_test_size, DIMENSIONS))

    # Now we swap test set and train set
    # 
    x_train = tf.placeholder(tf.float32, shape=( DIMENSIONS,_test_size,))
    y_train = tf.placeholder(tf.float32, shape=(DIMENSIONS,_test_size))
    x_test = tf.placeholder(tf.float32, shape=( DIMENSIONS,_train_size,))
    y_test = tf.placeholder(tf.float32, shape=(DIMENSIONS,_train_size))

    theta = tf.Variable(
        np.random.uniform(low=-0.1, high=0.1, size=(DIMENSIONS, DIMENSIONS)),
        dtype=np.float32, name='t_matrix')

    # theta = tf.Variable(
    #     saved_init(os.path.join('work','L6-1','20171029-202021-800','t_matrix.npy')),
    #     dtype=np.float32, name='t_matrix')


    # forward propagation
    # train_prediction = tf.matmul(x_train, theta)
    # test_prediction = tf.matmul(x_test, theta)
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
    print("initialized")
    print(theta.eval())
    for step in range(TRAINING_STEPS):
        _, train_c, test_c = s.run([optimizer, train_cost, test_cost],
                                   feed_dict={x_train: train_data,
                                              y_train: train_labels,
                                              x_test: test_data,
                                              y_test: test_labels})
        if step % 200 == 0:
            # it should return bias close to zero and parameters all close to 1 (see definition of f)
            print("\nAfter", step, "iterations:")
            # print("   Bias =", theta_0.eval(), ", Weights = ", theta.eval())
            print("   train cost =", train_c)
            print("   test cost =", test_c)
            time_str = time.strftime("%Y%m%d-%H%M%S")
            time_str+='-'
            time_str+=str(step)
            os.makedirs(os.path.join(work_dir, time_str))
            save_path = saver.save(s, os.path.join(work_dir, time_str, 'model.ckpt'))
            with open(os.path.join(work_dir, time_str,'status.txt'),'w') as fout:
                fout.write(str(train_c)+' '+str(test_c))
            np.save(os.path.join(work_dir, time_str,'t_matrix'),s.run('t_matrix:0'))
            print("Saved in",save_path)
