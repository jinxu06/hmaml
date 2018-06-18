import tensorflow as tf
import numpy as tf
from blocks.layers import conv2d, dense

def fc_nn(inputs):
    outputs = inputs
    num_hidden_layers = 2
    for i in range(num_hidden_layers):
        outputs = dense(outputs)
    return outputs

def conv_nn(inputs, num_channels):
    # outputs = tf.reshape(inputs, [-1, img_size, img_size, num_channels])
    outputs = inputs
    num_hidden_layers = 4
    for i in range(num_hidden_layers):
        outputs = conv2d(outputs)
    return outputs 
