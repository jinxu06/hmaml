import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
# plt.style.use("ggplot")

import data.mnist as mnist
from models.classifiers import MNISTClassifier

datasets = mnist.load(batch_size=1000)

from data.data_iterator import DataIterator

data_iter = DataIterator(train_set=datasets[0], val_set=datasets[1], test_set=datasets[2])

model = MNISTClassifier(num_classes=10, inputs=data_iter.next_op[0], targets=data_iter.next_op[1])

optimizer = tf.train.AdamOptimizer().minimize(model.loss)
global_init_op = tf.global_variables_initializer()

def train_epoch(model, optimizer, data_iter, metrics=["loss", "accuracy"]):
    ops = [model.evals[m] for m in metrics]
    ops += [optimizer]
    evals_sum = {m:0. for m in metrics}
    sess.run(data_iter.init_train_set_op)
    count = 0
    while True:
        try:
            *evals, _ = sess.run(ops)
            for i, m in enumerate(metrics):
                evals_sum[m] += evals[i]
            count += 1
        except tf.errors.OutOfRangeError:
            break
    evals_mean = {m:evals_sum[m] / count for m in metrics}
    return evals_mean

def eval_epoch(model, which_set, data_iter, metrics=["loss", "accuracy"]):
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
            evals = sess.run(ops)
            for i, m in enumerate(metrics):
                evals_sum[m] += evals[i]
            count += 1
        except tf.errors.OutOfRangeError:
            break
    print(evals_sum)
    print(count)
    evals_mean = {m:evals_sum[m] / count for m in metrics}
    return evals_mean

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(global_init_op)
    for k in range(100):
        print("epoch", k)
        evals = train_epoch(model, optimizer, data_iter)
        print(evals)
        if k%1 == 0:
            evals = eval_epoch(model, 'val', data_iter)
            print(evals)
