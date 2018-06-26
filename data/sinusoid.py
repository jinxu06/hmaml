import os
import random

from PIL import Image
import numpy as np

from .data_source import DataSource

class SinusoidDataSource(DataSource):

    def __init__(self, amp_range, phase_range, input_range):
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.input_range = input_range

    def _load(self):
        pass

    def split_train_test(self):
        pass

    def sample_classes(self, num_classes=1, which_set=None):
        num_funcs = num_classes
        amps = np.random.uniform(self.amp_range[0], self.amp_range[1], [num_funcs])
        phases = np.random.uniform(self.phase_range[0], self.phase_range[1], [num_funcs])
        for amp, phase in zip(amps, phases):
            yield SineWave(amp, phase, self.input_range)


class SineWave(object):
    """
    A single sine wave class.
    """
    def __init__(self, amp, phase, input_range):
        self.amp = amp
        self.phase = phase
        self.input_range = input_range
        self.tags = {"amp":amp, "phase":phase, "input_range":input_range}

    def sample(self, num_samples):
        inputs = np.random.uniform(self.input_range[0], self.input_range[1], [num_samples,1])
        outputs = self.amp * np.sin(inputs - self.phase)
        samples = np.concatenate([inputs, outputs], axis=-1)
        np.random.shuffle(samples)
        return samples
