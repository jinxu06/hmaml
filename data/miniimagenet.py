"""
Loading and using the Mini-ImageNet dataset.

To use these APIs, you should prepare a directory that
contains three sub-directories: train, test, and val.
Each of these three directories should contain one
sub-directory per WordNet ID.
"""

import os
import random

from PIL import Image
import numpy as np

def read_dataset(data_dir):
    """
    Read the Mini-ImageNet dataset.

    Args:
      data_dir: directory containing Mini-ImageNet.

    Returns:
      A tuple (train, val, test) of sequences of
        ImageNetClass instances.
    """
    return tuple(_read_classes(os.path.join(data_dir, x)) for x in ['train', 'val', 'test'])

def _read_classes(dir_path):
    """
    Read the WNID directories in a directory.
    """
    return [ImageNetClass(os.path.join(dir_path, f)) for f in os.listdir(dir_path)
            if f.startswith('n')]

# pylint: disable=R0903
class ImageNetClass:
    """
    A single image class.
    """
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self._cache = {}

    def sample(self, num_images):
        """
        Sample images (as numpy arrays) from the class.

        Returns:
          A sequence of 84x84x3 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.JPEG')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(name))
        return images

    def _read_image(self, name):
        if name in self._cache:
            return self._cache[name].astype('float32') / 0xff
        with open(os.path.join(self.dir_path, name), 'rb') as in_file:
            img = Image.open(in_file).resize((84, 84)).convert('RGB')
            self._cache[name] = np.array(img)
            return self._read_image(name)



# import csv
# import glob
# import os
# import shutil
# from PIL import Image
#
# data_dir = "/Users/Aaron-MAC/data/miniimagenet"
#
# def pre_processing():
#
#     print("unfinished, haven't downloaded imagenet yet")
#     quit()
#
#     for dataset in ['train', 'val', 'test']:
#         all_images = glob.glob(os.path.join(data_dir, dataset, "*", "*"))
#         for i, img_file in enumerate(all_images):
#             img = Image.open(img_file)
#             if not img.size == (84, 84):
#                 img = img.resize((84, 84), resample=Image.LANCZOS)
#                 img.save(img_file)
#         print(" *** finish processing ", dataset)
#
#     if not os.path.exists(os.path.join(data_dir, 'train')):
#         for dataset in ['train', 'val', 'test']:
#             os.makedirs(os.path.join(data_dir, dataset))
#             with open(os.path.join(data_dir, dataset + ".csv"), 'r') as f:
#                 reader = csv.reader(f, delimiter=",")
#                 last_label = ''
#                 next(reader)
#                 for i, row in emumerate(reader):
#                     label = row[1]
#                     img_name = row[0]
#                     if label != last_label:
#                         cur_dir = os.path.join(data_dir, dataset, label)
#                         os.makedirs(cur_dir)
#                         last_label = label
#                     shutil.move("images/"+img_name, cur_dir)
#     else:
#         print(" *** image folders already exist")
