"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow as tf
import components.losses as losses
from components.layers import conv2d, dense

DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)

class Model(object):

    def __init__(self, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self._model()
        self._loss()
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)

    def _model(self):
        pass

    def _loss(self):
        pass

class RegressionModel(Model):

    def __init__(self, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        super().__init__(optimizer=DEFAULT_OPTIMIZER, **optim_kwargs)

    def _loss(self):
        self.loss = losses.mse(pred=self.predictions, label=self.label_ph)

class ClassificationModel(Model):

    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.num_classes = num_classes
        super().__init__(optimizer=DEFAULT_OPTIMIZER, **optim_kwargs)
        self.predictions = tf.argmax(self.logits, axis=-1)

    def _loss(self):
        self.loss = losses.sparse_softmax_cross_entropy(pred=self.logits, label=self.label_ph)

class DensityEstimationModel(Model):

    def __init__(self, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        super().__init__(optimizer=DEFAULT_OPTIMIZER, **optim_kwargs)

    def _loss(self):
        self.loss = losses.logistic_mixture_loss(self.input_ph, self.dist_params)


class SinusoidModel(RegressionModel):

    def __init__(self, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 1))
        self.label_ph = tf.placeholder(tf.float32, shape=(None,))
        super().__init__(optimizer=DEFAULT_OPTIMIZER, **optim_kwargs)

    def _model(self):
        outputs = self.input_ph
        for _ in range(2):
            outputs = dense(outputs, 40, activation=tf.nn.relu)
        self.predictions = dense(outputs, 1, activation=None)


class OmniglotModel(ClassificationModel):
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        super().__init__(num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs)

    def _model(self):
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        for _ in range(4):
            out = conv2d(out, 64, 3, strides=2, padding='SAME', activation=tf.nn.relu, norm='batch_norm', is_training=True)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = dense(out, self.num_classes, activation=None)


class MiniImageNetModel(ClassificationModel):
    """
    A model for Mini-ImageNet classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 84, 84, 3))
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        super().__init__(num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs)

    def _model(self):
        out = self.input_ph
        for _ in range(4):
            out = conv2d(out, 32, 3, strides=1, padding='SAME', activation=None, norm='batch_norm', is_training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='SAME')
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = dense(out, self.num_classes, activation=None)
