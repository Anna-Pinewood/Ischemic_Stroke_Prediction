import random
from typing import Tuple
import os
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import make_scorer, fbeta_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
import torch
from math import floor
from torch.nn.modules.pooling import MaxPool2d, MaxPool3d


def maxpool_output_shape(input_size: Tuple[int, int],
                         layer: MaxPool2d):
    """Calculates the output shape of maxpool layer."""

    kernel_size = layer.kernel_size
    stride = layer.stride
    padding = layer.padding
    dilation = layer.dilation
    height, width = input_size

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    height_new = floor(
        (height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    width_new = floor(
        (width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return height_new, width_new


def maxpool_output_shape_3d(input_size: Tuple[int, int, int],
                            layer: MaxPool3d):
    """
    Calculates the output shape of a 3D maxpool layer.

    Args:
        img_heights (int): Height of each input image.
        img_width (int): Width of each input image.
        img_nums (int): Number of input images.
        pool_size (tuple of int): Size of the pooling window. Should be in the form (pool_height, pool_width, pool_depth).
        strides (tuple of int): Stride of the pooling window. Should be in the form (stride_height, stride_width, stride_depth).

    Returns:
        tuple of int: The output shape of the maxpool layer. Should be in the form (output_height, output_width, output_depth, img_nums).
    """
    kernel_size = layer.kernel_size
    stride = layer.stride
    padding = layer.padding
    dilation = layer.dilation

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 3
    if not isinstance(padding, tuple):
        padding = (padding, ) * 3
    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * 3
    if not isinstance(stride, tuple):
        stride = (stride, ) * 3

    d_in, h_in, w_in = input_size

    d_out = floor((d_in + 2*padding[0] - dilation[0] *
                   (kernel_size[0] - 1) - 1) / stride[0] + 1)
    h_out = floor((h_in + 2*padding[1] - dilation[1] *
                   (kernel_size[1] - 1) - 1) / stride[1] + 1)
    w_out = floor((w_in + 2*padding[2] - dilation[2] *
                   (kernel_size[2] - 1) - 1) / stride[2] + 1)
    return (d_out, h_out, w_out)


def show_tensor(tensor_img: torch.Tensor):
    img_norm = tensor_img.permute(1, 2, 0)[:, :, 0].detach().numpy()
    plt.imshow(img_norm, cmap='gray')
    plt.show()


def plot_tensors(tensors, labels=None):
    num_tensors = len(tensors)
    rows = 1
    cols = num_tensors
    _, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    for i in range(num_tensors):
        ax = axs[i] if num_tensors > 1 else axs
        img_norm = tensors[i].permute(1, 2, 0)[:, :, 0].detach().numpy()
        ax.imshow(img_norm, cmap='gray')
        ax.axis('off')
        if labels is not None:
            ax.set_title(labels[i])
        else:
            ax.set_title(f'Tensor {i+1}')

    plt.show()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
