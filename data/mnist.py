import tensorflow as tf
import numpy as np


def load(batch_size, split=[50000, 10000, 10000], one_hot=True):
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home="~/scikit_learn_data")
    images = mnist['data'].astype(np.float32)
    targets = mnist['target'].astype(np.int32)
    p = np.random.permutation(images.shape[0])
    images, targets = images[p], targets[p]
    datasets = []
    begin = 0
    for s in split:
        end = begin + s
        X = images[begin:end]
        y = targets[begin:end]
        print("y:", y.mean())
        X = tf.data.Dataset.from_tensor_slices(X)
        y = tf.data.Dataset.from_tensor_slices(y)
        if one_hot:
            y = y.map(lambda z: tf.one_hot(z, 10))
        dataset = tf.data.Dataset.zip((X, y)).shuffle(s).batch(batch_size)
        datasets.append(dataset)
        begin = end
    return datasets
