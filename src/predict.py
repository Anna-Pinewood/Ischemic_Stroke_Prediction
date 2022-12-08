import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.CTDataModule import CTDataModule
from neural_network import DeepSymNet

import sklearn
from sklearn.metrics import make_scorer, fbeta_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def plot_conf_matr(y_true, y_pred, nsamples: int, title: str = ''):
    """Built conf matrix and f1 and plot it."""
    y_true = pd.Series(y_true.numpy())
    y_pred = pd.Series(y_pred.numpy())

    matr = confusion_matrix(y_true, y_pred)
    matr_new = np.zeros((2,2))
    matr_new[0] = matr[0] / y_true.value_counts()[0]
    matr_new[1] = matr[1] / y_true.value_counts()[1]

    roc_auc = roc_auc_score(y_true, y_pred)
    fb_weighted = fbeta_score(y_true, y_pred, beta=1.1, average='weighted')

    sns.heatmap(matr_new, annot=True, fmt='.3', cmap='Blues')
    plt.title(f'f_beta_weighted={fb_weighted:.3}, \n {nsamples} сэмплов. {title}')
    plt.ylabel('Expert')
    plt.xlabel('Prediction')

if __name__ == "__main__":

    data_dir_new = '/home/high_fly_bird/Data/brain_dataset'
    dm = CTDataModule(data_dir=data_dir_new, batch_size=32)
    dm.setup('test')

    ckpt_path = '//home/high_fly_bird/Workspace/Ischemic_Stroke_Prediction/lightning_logs/version_4/checkpoints/epoch=50-step=153.ckpt'
    model = DeepSymNet.load_from_checkpoint(ckpt_path)


    trainer = pl.Trainer()
    predictions = trainer.test(model, datamodule=dm)

