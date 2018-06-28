import numpy as np
import tensorflow as tf

class DataIterator(object):

    def __init__(self, train_set, val_set=None, test_set=None):
        self.iterator = tf.data.Iterator.from_structure(train_set.output_types, train_set.output_shapes)
        self.next_op = self.iterator.get_next()
        self.init_train_set_op = self.iterator.make_initializer(train_set)
        if val_set is not None:
            self.init_val_set_op = self.iterator.make_initializer(val_set)
        if test_set is not None:
            self.init_test_set_op = self.iterator.make_initializer(test_set)
