import numpy as np
import os

class Dataset(object):
    def __init__(self, num_samples_per_class, batch_size):
        pass


class sinusoid(Dataset):
    def __init__(self):
        super().__init__()

class Omniglot(Dataset):

    def __init__(self):
        self.num_classes = 0
        self.img_size = 0
        self.data_dir = ""



class MiniImagenet(Dataset):
    pass
