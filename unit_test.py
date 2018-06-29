import numpy as np
import tensorflow as tf
from models.classifiers import MNISTClassifier
from components.learners import Learner
import data.mnist as mnist

datasets = mnist.load(data_dir="~/scikit_learn_data", num_classes=5, batch_size=100, split=[5./7, 1./7, 1./7])
dataset = datasets[0]
dataset.make_iterator(1)
val_dataset = datasets[1]
val_dataset.make_iterator(100)


model = MNISTClassifier(num_classes=5, inputs=None, targets=None)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(model.loss)

global_init_op = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(global_init_op)
    learner = Learner(session=sess, model=model)
    for epoch in range(20):
        for k in range(500):
            learner.one_shot_train_step(dataset, optimizer, batch_size=100, step_size=0.1)
        evals = learner.evaluate(val_dataset)
        print(evals)
