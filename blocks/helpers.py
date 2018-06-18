import numpy as np 
import tensorflow as tf

def int_shape(x):
    return list(map(int, x.get_shape()))

def get_name(layer_name, counters):
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name
