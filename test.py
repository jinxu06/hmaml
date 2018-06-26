import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import data.omniglot as omniglot
import data.sinusoid as sinusoid
import data.miniimagenet as miniimagenet


from data.load_data import Dataset


from data.omniglot import read_dataset, split_dataset, augment_dataset

from misc.plots import visualize_binary_images

DATA_DIR = "/Users/Aaron-MAC/Code/supervised-reptile/data/omniglot"

data_source = omniglot.OmniglotDataSource(data_dir=DATA_DIR)
dataset = Dataset(data_source=data_source, task_type='classification')

s = list(dataset.sample_mini_dataset(num_shots=10, num_classes=5))
images = []
for k in s:
    images.append(k[0])
images = np.array(images)
v = visualize_binary_images(images, layout=(5,10))
plt.imshow(v)
plt.show()

# data_source = sinusoid.SinusoidDataSource()
# dataset = Dataset(data_source=data_source)
#
# s = list(dataset.sample_mini_dataset(num_shots=200, task_type='regression'))
# s = np.array(s)
# inputs, labels = s[:, 0], s[:, 1]
# plt.scatter(inputs, labels)
#
# s = list(dataset.sample_mini_dataset(num_shots=200, task_type='regression'))
# s = np.array(s)
# inputs, labels = s[:, 0], s[:, 1]
# plt.scatter(inputs, labels)
#
# s = list(dataset.sample_mini_dataset(num_shots=200, task_type='regression'))
# s = np.array(s)
# inputs, labels = s[:, 0], s[:, 1]
# plt.scatter(inputs, labels)

# plt.show()

# s = g1.sample(200)
# plt.scatter(s[:,0], s[:,1])
# s = g2.sample(200)
# plt.scatter(s[:,0], s[:,1])
# plt.show()
