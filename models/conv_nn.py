import tensorflow as tf
import numpy as np
from blocks.loss import cross_entropy, mse
from components import fc_nn, conv_nn

class ConvNN(object):

    def __init__(self, counters={}):
        self.counters = counters

    def construct(self):
        self.__model()
        self.__loss()

    def __model(self, x, is_training):
        self.x = x
        self.is_training = is_training
        self.y = conv_nn(self.x)


    def __loss(self, type):
        self.loss_type = type
        if self.loss_type == 'mse'
            self.loss = None
        elif self.loss_type == 'cross_entropy':
            self.loss = None
