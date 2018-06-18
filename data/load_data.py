import numpy as np
import os

class Task(object):

    def __init__(self, task_type='classification', num_classes):
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

    def __init__(self, batch_size, meta_batch_size, rng=np.random.RandomState(1)):
        self.rng = rng
        self.batch_size = batch_size
        self.meta_batch_size = meta_batch_size

        self.train = None
        self.val = None
        self.test = None

    def sample_task(self, task_type='classification', dataset='train'):
        pass

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

    def sample_task(self, task_type='regression', dataset='train'):
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
        return init_inputs, outputs, amp, phase



class Omniglot(Dataset):

    def __init__(self):
        self.num_classes = 0
        self.img_size = 0
        self.data_dir = ""


class MiniImagenet(Dataset):
    pass
