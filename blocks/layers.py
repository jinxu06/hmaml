import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.helpers import int_shape, get_name

@add_arg_scope
def conv2d(inputs, num_filters, kernel_size, strides=1, padding='SAME', activation=None, norm="batch_norm", kernel_initializer=None, kernel_regularizer=None, is_training=False):
    """  conv2d
    norm: batch_norm | layer_norm | None
    """
    outputs = tf.layers.conv2d(inputs, num_filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    if norm == 'batch_norm':
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
    elif norm == 'layer_norm':
        outputs = tf.layers.layer_norm(outputs)

    if activation is not None:
        outputs = activation(outputs)
    print("    + conv2d", int_shape(inputs), int_shape(outputs), activation, norm)
    return outputs

@add_arg_scope
def dense(inputs, num_outputs, activation=None, norm=True, kernel_initializer=None, kernel_regularizer=None, is_training=False):
    """ dense
    norm: batch_norm | layer_norm | None
    """
    inputs_shape = int_shape(inputs)
    assert len(inputs_shape)==2, "inputs should be flattened first"
    outputs = tf.layers.dense(inputs, num_outputs, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    if norm == 'batch_norm':
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
    elif norm == 'layer_norm':
        outputs = tf.layers.layer_norm(outputs)
    if activation is not None:
        outputs = activation(outputs)
    print("    + dense", int_shape(inputs), int_shape(outputs), activation, norm)
    return outputs
