import os
import numpy

def pre_processing():
    pass

def generate_batch():
    amp_range = [0.1, 5.0]
    phase_range = [0, np.pi]
    input_range = [-5.0, 5.0]
    meta_batch_size = 10

    rng = None

    amp = rng.uniform(amp_range[0], amp_range[1], [meta_batch_size])
    phase = rng.uniform(phase_range[0], phase_range[1], [meta_batch_size])

    inputs = np.zeros([meta_batch_size, num_samples_per_class, 1])
    outputs = np.zeros([meta_batch_size, num_samples_per_class, 1])

    for idx in range(meta_batch_size):
        inputs[idx] = rng.uniform(input_range[0], input_range[1], [num_samples_per_class, 1])

        outputs[idx] = amp[idx] * np.sin(inputs[idx] - phase[idx])
    return inputs, outputs, amp, phase 
