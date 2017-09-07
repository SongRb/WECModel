import tensorflow as tf
import numpy as np
import platform
import time
import pickle
import time
import os

work_dir = os.path.join('data', 'train3')
dataset_name = 'dataset-yahoo-bus.pkl'

if platform.system().startswith('Windows'):
    print('Skipped word2vec processing')
    print('Loading from cached file')

    with open(os.path.join(work_dir,dataset_name),'rb') as fin:
        db = pickle.load(fin,encoding='latin1')
        train_x = db['x']
        train_y = db['y']
        train_w  =db['w']
else:
    try:
        import word2vec
    except ImportError:
        print('word2vec required. Install it via pip')

    with open('probTable.json','r') as fin:
        import json
        db = json.load(fin)

    model = word2vec.load('./text8.bin')
    vocab = [unicode(i) for i in model.vocab]
    vocab = set(vocab)
    train_set = list()
    missing_set = list()
    for pair in db:
        if pair[0] in vocab and pair[1] in vocab:
            train_set.append((model[pair[0]], model[pair[1]], float(pair[2])))
        else:
            missing_set.append(tuple(pair))

    train_x = list(); train_w = list(); train_y = list();
    for line in train_set:
        train_x.append(line[0])
        train_y.append(line[1])
        train_w.append(line[2])
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_w = np.array(train_w)

print('Successfully load train dataset')
print('Processing data with tensorflow...')

DIMENSIONS = len(train_x[0])
DS_SIZE = len(train_w)
ds = [(train_x[i], train_y[i]) for i in range(len(train_x))]
np.random.shuffle(ds)
TRAIN_RATIO = 0.6  # 60% of the dataset is used for training
_train_size = int(DS_SIZE * TRAIN_RATIO)
_test_size = DS_SIZE - _train_size
ALPHA = 1e-8  # learning rate
LAMBDA = 0.5  # L2 regularization factor
TRAINING_STEPS = 2000
train_data, train_labels = zip(*ds[0:_train_size])
test_data, test_labels = zip(*ds[_train_size:])

# modified
# train_labels = np.asarray(train_labels)
# test_labels = np.asarray(test_labels)
# train_labels = train_labels
# test_labels = test_labels

print('Dataset generated')


def cos_func(a, b):
    normalize_a = tf.nn.l2_normalize(a, 0)
    normalize_b = tf.nn.l2_normalize(b, 0)
    return tf.reduce_sum(tf.multiply(normalize_a, normalize_b))


# define the computational graph
print('Setting perferences')
graph = tf.Graph()
with graph.as_default():
    # declare graph inputs
    x_train = tf.placeholder(tf.float32, shape=(_train_size,DIMENSIONS))
    y_train = tf.placeholder(tf.float32, shape=(_train_size,DIMENSIONS))
    x_test = tf.placeholder(tf.float32, shape=(_test_size,DIMENSIONS))
    y_test = tf.placeholder(tf.float32, shape=(_test_size,DIMENSIONS))

    theta = tf.Variable(
        np.random.uniform(low=-0.01, high=0.01, size=(DIMENSIONS, DIMENSIONS)),
        dtype=np.float32, name='t_matrix')
    theta_0 = tf.Variable(
        [[0.0] for _ in range(DIMENSIONS)])  # don't forget the bias term!

    # forward propagation
    train_prediction = tf.matmul(x_train,theta)
    test_prediction = tf.matmul(x_test,theta)
    train_cost = tf.abs(cos_func(train_prediction, y_train))
    optimizer = tf.train.GradientDescentOptimizer(ALPHA).minimize(train_cost)
    test_cost = tf.abs(cos_func(test_prediction, y_test))

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
        if (step % 10 == 0):
            # it should return bias close to zero and parameters all close to 1 (see definition of f)
            print("\nAfter", step, "iterations:")
            # print("   Bias =", theta_0.eval(), ", Weights = ", theta.eval())
            print("   train cost =", train_c)
            print("   test cost =", test_c)
    time_str = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(work_dir, time_str))
    save_path = saver.save(s, os.path.join(work_dir, time_str, 'model.ckpt'))
    print(save_path)
	
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    np.save(os.path.join(work_dir, time_str,'t_matrix'),sess.run('t_matrix:0'))
