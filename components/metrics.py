import numpy as np
import tensorflow as tf

def num_correct(preds, labels):
    pass

def accuracy(preds, labels):
    pass

def mse_tf(preds, labels):
    preds = tf.reshape(preds, [-1])
    labels = tf.reshape(labels, [-1])
    return tf.reduce_mean(tf.square(preds-labels))

def mse(preds, labels):
    preds = np.reshape(preds, [-1])
    labels = np.reshape(labels, [-1])
    return np.mean((preds-labels)**2)

def nll(preds, dist_params):
    pass
