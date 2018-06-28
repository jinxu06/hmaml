import tensorflow as tf
import numpy as np


def load(batch_size, split=[50000, 10000, 10000], one_hot=True):
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home="~/scikit_learn_data")
    images = mnist['data'].astype(np.float32)
    targets = mnist['target'].astype(np.int32)
    datasets = []
    begin = 0
    for s in split:
        end = begin + s
        X = images[begin:end]
        y = targets[begin:end]
        X = tf.data.Dataset.from_tensor_slices(X)
        y = tf.data.Dataset.from_tensor_slices(y)
        if one_hot:
            y = y.map(lambda z: tf.one_hot(z, 10))
        dataset = tf.data.Dataset.zip((X, y)).shuffle(s).batch(batch_size)
        # iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        # dataset = iterator.make_initializer(dataset)
        datasets.append(dataset)
        begin = end
    return datasets
