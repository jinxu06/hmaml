import os
import numpy as np
import tensorflow as tf
import argparse
# import matplotlib.pyplot as plt
# plt.style.use("ggplot")

import data.mnist as mnist
import data.omniglot as omniglot
from models.classifiers import MNISTClassifier
from components.learners import Learner
from data.data_iterator import DataIterator


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--load_params', help='', action='store_true', default=False)
args = parser.parse_args()


if os.path.exists("/Users/Aaron-MAC/mldata/omniglot"):
    DATA_DIR = "/Users/Aaron-MAC/mldata/omniglot"
else:
    DATA_DIR = "/data/ziz/not-backed-up/jxu/omniglot"

train_meta_dataset, test_meta_dataset = omniglot.load(data_dir=DATA_DIR, inner_batch_size=5, num_train=1200, augment_train_set=False, one_hot=True)
datasets = train_meta_dataset.sample_mini_dataset(num_classes=5, num_shots=15, test_shots=5)
data_iter = DataIterator(train_set=datasets[0])
model = MNISTClassifier(num_classes=5, inputs=data_iter.next_op[0], targets=data_iter.next_op[1])

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(model.loss)
global_init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
save_dir = "/data/ziz/jxu/hmaml-saved-models"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if args.load_params:
        ckpt_file = save_dir + '/params_' + "mnist" + '.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, ckpt_file)

    learner = Learner(session=sess, model=model)
    sess.run(global_init_op)

    results = []

    for k in range(20):
        print(k, "......")
        datasets = train_meta_dataset.sample_mini_dataset(num_classes=5, num_shots=15, test_shots=5)
        data_iter.reinitialize(train_set=datasets[0], val_set=None, test_set=datasets[1])
        r = learner.evaluate(data_iter, model.evals["accuracy"], optimizer, num_iter=5)
        results.append(r)
    print(np.array(results).mean(0))
