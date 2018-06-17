import numpy as np
import os

class Dataset(object):

    def __init__(self, batch_size, meta_batch_size, rng=np.random.RandomState(1)):
        self.rng = rng
        self.batch_size = batch_size
        self.meta_batch_size = meta_batch_size

    def train(self, shuffle=False, limit=-1):
        pass

    def val(self, shuffle=False, limit=-1):
        pass

    def test(self.shuffle=False, limit=-1):
        pass


class Sinusoid(Dataset):

    def __init__(self):
        super().__init__()
        self.amp_range = [0.1, 5.0]
        self.phase_range = [0, np.pi]
        self.input_range = [-5.0, 5.0]
        self.dim_inputs = 1
        self.dim_outputs = 1

    def meta_dataset_split(self):
        meta_train_size, meta_val_size = 1200, 100


class Omniglot(Dataset):

    def __init__(self):
        self.num_classes = 0
        self.img_size = 0
        self.data_dir = ""

    def meta_dataset_split(self):




class MiniImagenet(Dataset):
    pass
