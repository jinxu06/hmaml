import numpy as np
import os

class Task(object):

    def __init__(self, task_type='classification', num_classes=1):
        """ task_type: classification | regression | density_estimation
        """
        self.task_type = task_type
        self.num_classes = num_classes
        self.train_set = None   # Iterator
        self.val_set = None
        self.test_set = None

    def sample(self, dataset='train'):
        pass

class Dataset(object):

    def __init__(self, meta_batch_size, num_samples_per_class, dim_input, dim_output, task_type, rng=np.random.RandomState(1)):
        # hyperparameters
        self.rng = rng
        self.meta_batch_size = meta_batch_size
        self.num_samples_per_class = num_samples_per_class
        # task descriptions
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.task_type = task_type
        # datasets
        self.train = None
        self.val = None
        self.test = None

    def sample_batch_of_tasks(self, dataset='train'):
        pass


class Sinusoid(Dataset):

    def __init__(self, amp_range, phase_range, input_range, meta_batch_size, num_samples_per_class, rng=np.random.RandomState(1)):
        super().__init__(meta_batch_size, num_samples_per_class, 1, 1, 'regression', rng)
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.input_range = input_range

    def sample_batch_of_tasks(self, dataset='train'):

        amp_batch = self.rng.uniform(self.amp_range[0], self.amp_range[1], [self.meta_batch_size])
        phase_batch = self.rng.uniform(self.phase_range[0], self.phase_range[1], [self.meta_batch_size])

        dim_input, dim_output = 1, 1
        inputs = np.zeros([self.meta_batch_size, self.num_samples_per_class, dim_input])
        outputs = np.zeros([self.meta_batch_size, self.num_samples_per_class, dim_output])

        for b in range(self.meta_batch_size):
            inputs[b] = self.rng.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            outputs[b] = amp_batch[b] * np.sin(inputs[b]-phase_batch[b])
        return inputs, outputs, (amp_batch, phase_batch)


class Omniglot(Dataset):

    def __init__(self, meta_batch_size, num_samples_per_class, num_classes, rng=np.random.RandomState(1)):
        super().__init__(meta_batch_size, num_samples_per_class, 28*28, 5,num_classes, rng)
        self.num_classes = num_classes

    def sample_batch_of_tasks(self, dataset='train'):


class MiniImagenet(Dataset):
    pass
