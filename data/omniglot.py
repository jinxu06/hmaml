"""
Loading and augmenting the Omniglot dataset.
To use these APIs, you should prepare a directory that
contains all of the alphabets from both images_background
and images_evaluation.
"""

import os
import random

from PIL import Image
import numpy as np

from .data_source import DataSource


class OmniglotDataSource(DataSource):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = None
        
    def _load(self):
        self.data = read_dataset(self.data_dir)

    def split_train_test(self, num_train, augment_train_set=True):
        if self.data is None:
            self._load()
        self.train_set, self.test_set = split_dataset(self.data)
        if augment_train_set:
            self.train_set = list(augment_dataset(self.train_set))
        self.test_set = list(self.test_set)

    def sample_classes(self, num_classes, which_set='train'):
        if which_set == 'train':
            dataset = self.train_set
        elif which_set == 'test':
            dataset = self.test_set
        else:
            raise Exception('which_set is either train or test')
        shuffled = list(dataset)
        random.shuffle(shuffled)
        return shuffled[:num_classes]


def read_dataset(data_dir):
    """
    Iterate over the characters in a data directory.
    Args:
      data_dir: a directory of alphabet directories.
    Returns:
      An iterable over Characters.
    The dataset is unaugmented and not split up into
    training and test sets.
    """
    for alphabet_name in sorted(os.listdir(data_dir)):
        alphabet_dir = os.path.join(data_dir, alphabet_name)
        if not os.path.isdir(alphabet_dir):
            continue
        for char_name in sorted(os.listdir(alphabet_dir)):
            if not char_name.startswith('character'):
                continue
            yield Character(os.path.join(alphabet_dir, char_name), 0, tags={'alphabet': alphabet_name, 'character': char_name, 'rotation': 0})

def split_dataset(dataset, num_train=1200):
    """
    Split the dataset into a training and test set.
    Args:
      dataset: an iterable of Characters.
    Returns:
      A tuple (train, test) of Character sequences.
    """
    all_data = list(dataset)
    random.shuffle(all_data)
    return all_data[:num_train], all_data[num_train:]

def augment_dataset(dataset):
    """
    Augment the dataset by adding 90 degree rotations.
    Args:
      dataset: an iterable of Characters.
    Returns:
      An iterable of augmented Characters.
    """
    for character in dataset:
        for rotation in [0, 90, 180, 270]:
            tags = character.tags
            tags.update({'rotation': rotation})
            yield Character(character.dir_path, rotation=rotation, tags=tags)


class Character:
    """
    A single character class.
    """
    def __init__(self, dir_path, rotation=0, tags={}):
        self.dir_path = dir_path
        self.rotation = rotation
        self._cache = {}
        self.tags = tags.copy()

    def sample(self, num_images):
        """
        Sample images (as numpy arrays) from the class.
        Returns:
          A sequence of 28x28 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.png')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(os.path.join(self.dir_path, name)))
        return images

    def _read_image(self, path):
        if path in self._cache:
            return self._cache[path]
        with open(path, 'rb') as in_file:
            img = Image.open(in_file).resize((28, 28)).rotate(self.rotation)
            self._cache[path] = np.array(img).astype('float32')
            return self._cache[path]

# from PIL import Image
# import glob
# import os
#
# data_dir = "/Users/Aaron-MAC/data/omniglot"
#
# def pre_processing():
#     for dataset in ['images_background', "images_evaluation"]:
#         all_images = glob.glob(os.path.join(data_dir, dataset, "*", "*", "*"))
#         for i, img_file in enumerate(all_images):
#             img = Image.open(img_file)
#             if not img.size == (28, 28):
#                 img = img.resize((28, 28), resample=Image.LANCZOS)
#                 img.save(img_file)
#         print(" *** finish processing ", dataset)
