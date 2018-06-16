from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf

class MAML(object):

    def __init__(self, dim_inputs, dim_outputs, test_num_updates):

        self.dim_inputs = dim_inputs
        self.dim_outputs = dim_outputs
        self.update_lr = 0.1
        self.meat_lr = 0.1
        self.test_num_updates = test_num_updates

        self.model = None
        self.loss = None
