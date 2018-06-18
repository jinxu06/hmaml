import tensorflow as tf


def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def cross_entropy(pred, label):
    bsize = int_shape(pred)[0]
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / bsize
