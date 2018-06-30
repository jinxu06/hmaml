import numpy as np
import tensorflow as tf
from models.classifiers import MNISTClassifier
from components.learners import Learner
import data.mnist as mnist

meta_train_set, meta_val_set, meta_test_set = mnist.load(data_dir="~/scikit_learn_data", num_classes=5, batch_size=5, split=[5./7, 1./7, 1./7], return_meta=True)
train_set, test_set = meta_train_set.sample_mini_dataset(num_classes=5, num_shots=1000, test_shots=5, classes=[5,6,7,8,9])


model = MNISTClassifier(num_classes=5, inputs=None, targets=None)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(model.loss)

global_init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
save_dir = "/data/ziz/jxu/hmaml-saved-models"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(global_init_op)

    # ckpt_file = save_dir + '/params_' + "mnist" + '.ckpt'
    # print('restoring parameters from', ckpt_file)
    # saver.restore(sess, ckpt_file)

    learner = Learner(session=sess, model=model)
    for epoch in range(20):
        print(epoch, "......")
        learner.train(train_set, optimizer)
        evals = learner.evaluate(test_set)
        print(evals)

    # saver.save(sess, save_dir + '/params_' + "mnist" + '.ckpt')
