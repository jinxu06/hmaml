import numpy as np
import tensorflow as tf

def int_shape(x):
    s = x.get_shape()
    if s[0] is None:
        s[0] = -1
    return list(map(int, s))

def get_name(layer_name, counters):
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name
