import os
import random

from PIL import Image
import numpy as np

def read_dataset(num_funcs, amp_range=[0.1, 5.0], phase_range=[0, np.pi], input_range=[-5.0, 5.0], rng=np.random.RandomState(0)):
    amps = rng.uniform(amp_range[0], amp_range[1], [num_funcs])
    phases = rng.uniform(phase_range[0], phase_range[1], [num_funcs])
    for amp, phase in zip(amps, phases):
        yield SineWave(amp, phase, input_range, rng)

def split_dataset(dataset, num_train):
    all_data = list(dataset)
    random.shuffle(all_data)
    return all_data[:num_train], all_data[num_train:]


class SineWave(object):
    """
    A single sine wave class.
    """
    def __init__(self, amp, phase, input_range, rng=np.random.RandomState(0)):
        self.amp = amp
        self.phase = phase
        self.input_range = input_range
        self.rng = rng

    def sample(self, num_samples):
        inputs = self.rng.uniform(self.input_range[0], self.input_range[1], [num_samples,1])
        outputs = self.amp * np.sin(inputs - self.phase)
        samples = np.concatenate([inputs, outputs], axis=-1)
        self.rng.shuffle(samples)
        return samples


# import os
# import numpy
#
# def pre_processing():
#     pass
#
# def generate_batch():
#     amp_range = [0.1, 5.0]
#     phase_range = [0, np.pi]
#     input_range = [-5.0, 5.0]
#     meta_batch_size = 10
#
#     rng = None
#
#     amp = rng.uniform(amp_range[0], amp_range[1], [meta_batch_size])
#     phase = rng.uniform(phase_range[0], phase_range[1], [meta_batch_size])
#
#     inputs = np.zeros([meta_batch_size, num_samples_per_class, 1])
#     outputs = np.zeros([meta_batch_size, num_samples_per_class, 1])
#
#     for idx in range(meta_batch_size):
#         inputs[idx] = rng.uniform(input_range[0], input_range[1], [num_samples_per_class, 1])
#
#         outputs[idx] = amp[idx] * np.sin(inputs[idx] - phase[idx])
#     return inputs, outputs, amp, phase
