import os
import random

from PIL import Image
import numpy as np

class SineWave(object):
    """
    A single sine wave class.
    """
    def __init__(self, amp, phase):
        self.amp = amp
        self.phase = phase

    def sample(self, num_inputs):
        """
        Sample SineWave (as numpy arrays) from the class.

        Returns:
          ...
        """


        names = [f for f in os.listdir(self.dir_path) if f.endswith('.JPEG')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(name))
        return images

    def _generate_function(self, amp, phase):
        pass


    def _read_image(self, name):
        if name in self._cache:
            return self._cache[name].astype('float32') / 0xff
        with open(os.path.join(self.dir_path, name), 'rb') as in_file:
            img = Image.open(in_file).resize((84, 84)).convert('RGB')
            self._cache[name] = np.array(img)
            return self._read_image(name)






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
