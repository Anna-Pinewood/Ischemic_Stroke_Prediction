import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.core.module import LightningModule
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import mean_squared_error

class ConvNormPool(nn.Module):
    """Conv Skip-connection module"""
    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        norm_type='bachnorm'
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_3 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )

        if norm_type == 'group':
            self.normalization_1 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
            self.normalization_2 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
            self.normalization_3 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
        else:
            self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_3 = nn.BatchNorm1d(num_features=hidden_size)

    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = F.silu(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = F.silu(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        conv3 = self.conv_3(x)
        x = self.normalization_3(conv1+conv3)
        x = F.silu(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        x = nn.MaxPool1d(kernel_size=2)(x)
        return x
class RNNModel(LightningModule):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers=1, lr=0.001):
        super().__init__()
        print("Hallo!")
        self.n_classes = 5
        self.lr = lr
        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=hidden_size,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
        )
        self.rnn_layer = nn.RNN(input_size=46, hidden_size=hidden_size, num_layers=num_layers, dropout=0.1, bidirectional=True)

        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(in_features=hidden_size, out_features=self.n_classes)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x, _ = self.rnn_layer(x)
        x = self.avgpool(x)
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=1)#.squeeze(1)
        return x
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("training_loss", loss, on_epoch=True,
                 on_step=True, prog_bar=True)
        return loss
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        # self.log('rmse', loss, on_epoch=True, prog_bar=True)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        rmse = np.sqrt(mean_squared_error(y, y_hat.detach().numpy()))
        # self.log('roc_auc', roc_auc)
        self.log('rmse', rmse)
        return {"rmse": rmse,
                "y_true": y, "y_pred": y_hat}


  # nn.CrossEntropyLoss()