from math import ceil
from matplotlib import pyplot as plt
import numpy as np
from pyparsing import Optional
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import torch

import seaborn as sns

import pandas as pd
from tqdm import tqdm

from src.CTDataModule import CTDataModule
from src.neural_network import DeepSymNet


def get_test_predictions(dm_predict: CTDataModule,
                         model: DeepSymNet):
    """Get predictions from model test_step."""
    dm_predict.setup('test')
    dataloader = dm_predict.test_dataloader()

    n_images = dm_predict.n_images
    n_batches = ceil(n_images / dm_predict.batch_size)

    y_true_all = np.array([])
    y_pred_all = np.array([])

    for i in tqdm(range(n_batches)):
        batch = next(iter(dataloader))
        test_step_output = model.test_step(batch, i)
        y_true = test_step_output['y_true'].numpy()
        y_pred = test_step_output['y_pred'].detach().numpy()
        y_true_all = np.append(y_true_all, y_true)
        y_pred_all = np.append(y_pred_all, y_pred)

    result = pd.DataFrame({'y_true': y_true_all, 'y_pred': y_pred_all})

    return result


def plot_conf_matrix(y_true: pd.Series,
                     y_pred: pd.Series):
    """Build conf matrix and plot it."""

    n_samples = len(y_true)

    matrix = confusion_matrix(y_true, y_pred, normalize='true')

    plot = sns.heatmap(matrix, annot=True, fmt='.3', cmap='Blues')
    plot.set(xlabel='True', ylabel='Pred')
    plot.set(title=f'{n_samples} сэмлов')
    plt.show()
    return plot


def plot_roc_curve(y_pred_proba, y_true):
    roc_auc_metric = roc_auc_score(y_true, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_true,  y_pred_proba)

    plt.plot(fpr, tpr)
    plt.title(f'roc_auc = {roc_auc_metric}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
