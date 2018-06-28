import numpy as np
import tensorflow as tf
from components.layers import conv2d, dense

class MNISTClassifier(object):

    def __init__(self, num_classes, inputs=None, targets=None):
        self.num_classes = num_classes
        if inputs is None:
            inputs = tf.placeholder(tf.float32, shape=(None, 28, 28))
        if targets is None:
            targets = tf.placeholder(tf.int32, shape=(None, 10))
        self.inputs, self.targets = inputs, targets
        self.outputs = self._model(self.inputs)
        self.loss = self._loss(self.outputs, self.targets)
        self.evals = self._eval(self.outputs, self.targets)

    def _model(self, inputs):
        out = tf.reshape(inputs, (-1, 28, 28, 1))
        for _ in range(4):
            out = conv2d(out, 64, 3, strides=2, padding='SAME', activation=tf.nn.relu, norm='batch_norm', is_training=True)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        out = dense(out, self.num_classes, activation=None, norm='None')
        return out

    def _loss(self, outputs, targets):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=outputs))

    def _eval(self, outputs, targets):
        preds = tf.argmax(outputs, 1)
        labels = tf.argmax(targets, 1)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
        return {"loss":self.loss, "accuracy":acc}
