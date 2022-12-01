from typing import Tuple

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
    height_new = floor((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    width_new = floor((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return height_new, width_new


def plot_conf_matr(y_true, y_pred, nsamples: int, title: str = ''):
    """Built conf matrix and f1 and plot it."""
    y_true = pd.Series(y_true.numpy())
    y_pred = pd.Series(y_pred.numpy())

    matr = confusion_matrix(y_true, y_pred)
    matr_new = np.zeros((2, 2))
    matr_new[0] = matr[0] / y_true.value_counts()[0]
    matr_new[1] = matr[1] / y_true.value_counts()[1]

    roc_auc = roc_auc_score(y_true, y_pred)
    fb_weighted = fbeta_score(y_true, y_pred, beta=1.1, average='weighted')
    sns.heatmap(matr_new, annot=True, fmt='.3', cmap='Blues')

    plt.title(f'f_beta_weighted={fb_weighted:.3}, \n {nsamples} сэмплов. {title}')
    plt.ylabel('Expert')
    plt.xlabel('Prediction')


def show_tensor(tensor_img: torch.Tensor):
    img_norm = tensor_img.permute(1, 2, 0)[:, :, 0].detach().numpy()
    plt.imshow(img_norm, cmap='gray')
    plt.show()
