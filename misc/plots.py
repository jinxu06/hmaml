import numpy as np
import matplotlib.pyplot plt
import seaborn as sns
from PIL import Image

def _tile_images(imgs, size):
    """
    imgs: numpy array with shape (batch_size, image height, image width, num_channels)
    size: integer or tuple (len 2)
    """
    if isinstance(size, int):
        size = (size, size)
    imgs = imgs[:size[0]*size[1], :, :, :]
    img_h, img_w = imgs.shape[1], imgs.shape[2]
    all_images = np.zeros((img_h*size[0], img_w*size[1], 3), np.uint8)
    for j in range(size[0]):
        for i in range(size[1]):
            all_images[img_h*j:img_h*(j+1), img_w*i:img_w*(i+1), :] = imgs[j*size[1]+i, :, :, :]
    return all_images

def visualize_rgb_images(images, save_path=None, layout=[5,5], value_range=[-1., 1.]):
    images = (images - value_range[0]) / (value_range[1]-value_range[0]) * 255.
    images = np.rint(images).astype(np.uint8)
    view = _tile_images(images, size=layout)
    if save_path is None:
        return view
    view = Image.fromarray(view, 'RGB')
    view.save(save_path)

def visualize_binary_images(images, save_path=None, layout=[5,5]):
    "untest"
    images = np.rint(images).astype(np.bool)
    view = _tile_images(images, size=layout)
    if save_path is None:
        return view
    view = Image.fromarray(view, '1')
    view.save(save_path)
