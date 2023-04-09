import os

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
import random


def gauss_noise_tensor(img):
    """Add Gaussian noise to image."""
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    sigma = 1.0

    out = img + sigma * torch.randn_like(img)

    if out.dtype != dtype:
        out = out.to(dtype)

    return out


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def random_sharpness_or_blur(img):
    """Randomly make image more shurp or blurry."""
    rand_trans = [transforms.RandomAdjustSharpness(
        sharpness_factor=2), transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1))]
    trans_idx = random.randint(0, 1)
    trans = rand_trans[trans_idx]
    return trans(img)


def crop_image(img: np.ndarray, tolerance=70) -> Image.Image:
    """
    Crops black borders of image.
    Parameters
    ----------
    img: Image to crop
    tolerance: how hard would black borders be cropped.
    If 0 - no cropping.
    """
    mask = img > tolerance
    cropped = img[np.ix_(mask.any(1), mask.any(0))]
    PIL_image = Image.fromarray(cropped)
    return PIL_image
