import numpy as np
import tensorflow as tf
import os

# from data.load_data import Sinusoid
#

# dgen = Sinusoid(amp_range=[0.1, 5.0], phase_range=[0, np.pi], input_range=[-5., 5.], meta_batch_size=5, num_samples_per_class=10)
# inputs, outputs, _ = dgen.sample_batch_of_tasks()
# print(inputs)
# print(outputs)


DATA_DIR = "/Users/Aaron-MAC/Code/supervised-reptile/data"

omniglot_dir = os.path.join(DATA_DIR, "omniglot")

from data.omniglot import *

rt = read_dataset(omniglot_dir)

for item in rt:
    print(item)
