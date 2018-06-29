import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
# plt.style.use("ggplot")

import data.mnist as mnist
from models.classifiers import MNISTClassifier

datasets = mnist.load(data_dir="~/scikit_learn_data", num_classes=5, batch_size=100, split=[5./7, 1./7, 1./7])

from data.data_iterator import DataIterator

data_iter = DataIterator(train_set=datasets[0], val_set=datasets[1], test_set=datasets[2])

model = MNISTClassifier(num_classes=10, inputs=data_iter.next_op[0], targets=data_iter.next_op[1])

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer().minimize(model.loss)
global_init_op = tf.global_variables_initializer()

def train_epoch(sess, model, optimizer, data_iter, metrics=["loss", "accuracy"]):
    ops = [model.evals[m] for m in metrics]
    ops += [optimizer]
    evals_sum = {m:0. for m in metrics}
    sess.run(data_iter.init_train_set_op)
    count = 0
    while True:
        try:
            *evals, _ = sess.run(ops, feed_dict={model.is_training: True})
            for i, m in enumerate(metrics):
                evals_sum[m] += evals[i]
            count += 1
        except tf.errors.OutOfRangeError:
            break
    evals_mean = {m:evals_sum[m] / count for m in metrics}
    return evals_mean

def eval_epoch(sess, model, which_set, data_iter, metrics=["loss", "accuracy"]):
    ops = [model.evals[m] for m in metrics]
    evals_sum = {m:0. for m in metrics}
    if which_set == 'train':
        sess.run(data_iter.init_train_set_op)
    elif which_set == 'val':
        sess.run(data_iter.init_val_set_op)
    elif which_set == 'test':
        sess.run(data_iter.init_test_set_op)
    count = 0
    while True:
        try:
            evals = sess.run(ops, feed_dict={model.is_training: False})
            for i, m in enumerate(metrics):
                evals_sum[m] += evals[i]
            count += 1
        except tf.errors.OutOfRangeError:
            break
    evals_mean = {m:evals_sum[m] / count for m in metrics}
    return evals_mean

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    num_epoches = 50
    sess.run(global_init_op)
    for k in range(num_epoches+1):
        print("epoch", k)
        evals = train_epoch(sess, model, optimizer, data_iter)
        print(evals)
        if k%5 == 0:
            evals = eval_epoch(sess, model, 'train', data_iter)
            print(evals)
            evals = eval_epoch(sess, model, 'val', data_iter)
            print(evals)
            evals = eval_epoch(sess, model, 'test', data_iter)
            print(evals)
