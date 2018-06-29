import os 
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
# plt.style.use("ggplot")

import data.mnist as mnist
import data.omniglot as omniglot
from models.classifiers import MNISTClassifier

if os.path.exists("/Users/Aaron-MAC/mldata/omniglot"):
    DATA_DIR = "/Users/Aaron-MAC/mldata/omniglot"
else:
    DATA_DIR = "/data/ziz/not-backed-up/jxu/omniglot"

train_meta_dataset, test_meta_dataset = omniglot.load(data_dir=DATA_DIR, inner_batch_size=5, num_train=1200, augment_train_set=False, one_hot=True)
datasets = train_meta_dataset.sample_mini_dataset(num_classes=5, num_shots=15, test_shots=5)

from data.data_iterator import DataIterator

data_iter = DataIterator(train_set=datasets[0], val_set=None, test_set=datasets[1])

model = MNISTClassifier(num_classes=5, inputs=data_iter.next_op[0], targets=data_iter.next_op[1])

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(model.loss)
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

saver = tf.train.Saver()
load_params = True
save_dir = "/data/ziz/jxu/hmaml-saved-models"


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if load_params:
        ckpt_file = save_dir + '/params_' + "mnist" + '.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, ckpt_file)

    num_epoches = 2
    sess.run(global_init_op)
    for k in range(num_epoches+1):
        print("epoch", k)
        evals = train_epoch(sess, model, optimizer, data_iter)
        print(evals)
        if k%1 == 0:
            evals = eval_epoch(sess, model, 'train', data_iter)
            print(evals)
            # evals = eval_epoch(sess, model, 'val', data_iter)
            # print(evals)
            evals = eval_epoch(sess, model, 'test', data_iter)
            print(evals)
