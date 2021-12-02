from skimage.morphology import skeletonize
import torch
import numpy as np
from scipy.signal import convolve2d


def thin_kernal_count(image):
    # given 2d binary tensor mask
    # to numpy and skeletonize
    image = torch.squeeze(image)
    image = skeletonize(image.numpy())
    # apply 3x3 kernal sum
    kernalized_image = convolve2d(image, np.ones((3,3), dtype=np.int8), mode='same', boundary='fill') == 2
    head_tail = np.logical_and(image, kernalized_image)
    # entry with value 2 is head or tail
    count = head_tail.sum() // 2

    return count
