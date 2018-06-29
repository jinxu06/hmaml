import numpy as np
import tensorflow as tf

def int_shape(x):
    s = x.get_shape()
    if isinstance(s[0], int):
        return list(map(int, s))
    else:
        return [-1] + list(map(int, s[1:]))

def get_name(layer_name, counters):
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

def one_hot(y, num_classes):
    y = np.array(y).astype(np.int32)
    r = np.zeros((len(y), num_classes))
    r[np.arange(len(y)), y] = 1 
    return r
