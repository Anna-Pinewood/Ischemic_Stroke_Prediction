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
from torch.nn.modules.pooling import MaxPool2d


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
