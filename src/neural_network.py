import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.core.module import LightningModule
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error

from src.utils import maxpool_output_shape

LOGGER = logging.getLogger()

IMG_HEIGHT = 128
IMG_WIDTH = 98


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_chanels)

    def forward(self, x):
        return nn.functional.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_1x1: int,
                 reduce_3x3: int,
                 out_3x3: int,
                 reduce_5x5: int,
                 out_5x5: int,
                 out_pooling: int):
        super().__init__()
        self.branch_1filter = ConvBlock(in_channels, out_1x1, kernel_size=1)

        self.branch_3filter = nn.Sequential(ConvBlock(in_channels, reduce_3x3, kernel_size=1),
                                            ConvBlock(
                                                reduce_3x3, out_3x3, kernel_size=3, padding=1)
                                            )
        self.branch_5filter = nn.Sequential(
            ConvBlock(in_channels, reduce_5x5, kernel_size=1),
            ConvBlock(reduce_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch_maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_pooling, kernel_size=1),
        )

    def forward(self, x):
        branches = (self.branch_1filter, self.branch_3filter,
                    self.branch_5filter, self.branch_maxpool)
        output = torch.cat([branch(x) for branch in branches], 1)
        return output


class SiameseAndDifferenceBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.inception_chain = nn.Sequential(
            InceptionBlock(in_channels=1,
                           out_1x1=64,
                           reduce_3x3=64,
                           out_3x3=64,
                           reduce_5x5=64,
                           out_5x5=64,
                           out_pooling=64),
            InceptionBlock(256, 64, 64, 64, 64, 64, 64),
            InceptionBlock(256, 64, 64, 64, 64, 64, 64),
            InceptionBlock(256, 64, 64, 64, 64, 64, 64)
        )

    def forward(self, x):
        input1 = self._split_tensor(x, width=IMG_WIDTH)
        input2 = self._split_tensor(x, left=False, width=IMG_WIDTH)
        output1 = self.inception_chain(input1)
        output2 = self.inception_chain(input2)
        output2_reflected = torch.flip(output2, dims=[3])

        return abs(output1 - output2_reflected)

    @staticmethod
    def _split_tensor(
            batch: torch.Tensor,
            left: bool = True,
            width: int = IMG_WIDTH):
        """
        Splits input batch in half for Siamese Network.
        Parameters
        ----------
        batch: torch.Tensor
            Output of next(iter(dm_loader))[0] of shape
            torch.Size([ n_samples_in_batch, 1, height, width])
        left: bool
            Whether to return left part of brain or not.
        width: int
            The width of the images along which the split is made.
        Returns
        -------
        torch.Tensor
            Batch, each sample is cut in half.
        """
        return batch[:, :, :, :(width // 2)] if left \
            else batch[:, :, :, (width // 2):]


class DeepSymNet(LightningModule):
    def __init__(self, learning_rate: float = 1e-5):
        """
        Initialize the model with specific learning rate.
        Parameters
        ----------
        learning_rate: float
            Argument of torch.optim.Adam of configure_optimizers()
            torch.optim.Adam(self.parameters(), learning_rate=self.learning_rate)
        """
        super().__init__()

        self.siamese_part = SiameseAndDifferenceBlock()

        self.shared_tunnel = nn.Sequential(
            InceptionBlock(256, 64, 64, 64, 64, 64, 64),
            InceptionBlock(256, 64, 64, 64, 64, 64, 64),
            nn.MaxPool2d(kernel_size=3, padding=1))

        height_new, width_new = maxpool_output_shape(
            input_size=(IMG_HEIGHT, IMG_WIDTH//2),
            layer=self.shared_tunnel[-1])
        units_fc = height_new * width_new * 256

        self.fc1 = nn.Sequential(
            nn.Linear(units_fc, 1)  # units_fc TODO
        )
        self.threshold = None
        self.learning_rate = learning_rate

    def forward(self, x):
        LOGGER.debug(f'Input shape is {x.shape}')
        output_siam = self.siamese_part(x)
        LOGGER.debug(
            f'Output of SiameseAndDifferenceBlock block is {output_siam.shape}')
        output = self.shared_tunnel(output_siam)
        LOGGER.debug(f'Output of IM-IM-MP block is {output.shape}')

        output = output.view(output.size()[0], -1)
        LOGGER.debug(f'Reshaping for fully connected to {output.shape}')
        output = self.fc1(output)
        output = nn.Sigmoid()(output)
        LOGGER.debug(f'Output shape is {output.shape}')
        output = output.reshape(output.shape[0])

        LOGGER.debug(f'Output reshaped to {output.shape}')
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def binary_cross_entropy_loss(self, y_predicted, y_true):
        y_true = y_true.to(torch.float32)
        LOGGER.debug(f'y_pred \n {y_predicted}')
        LOGGER.debug(f'y_true \n {y_true}')
        loss = nn.BCELoss()(y_predicted, y_true)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.binary_cross_entropy_loss(y_hat, y)
        self.log("training_loss", loss, on_epoch=True,
                 on_step=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = self.binary_cross_entropy_loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        if len(torch.unique(y)) < 2:
            LOGGER.warning('Only one class present in y_true.'
                           'ROC AUC score is not defined in that case.')
            roc_auc = -1
        else:
            roc_auc = roc_auc_score(y, y_hat.detach().numpy())
        rmse = np.sqrt(mean_squared_error(y, y_hat.detach().numpy()))
        self.log('roc_auc', roc_auc)
        self.log('rmse', rmse)
        return {"roc_auc":  roc_auc, "rmse": rmse,
                "y_true": y, "y_pred": y_hat}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        return {'logits': y, 'labels': y_hat}

    @staticmethod
    def find_threshold(y_predicted, y_true,
                       metric: Callable = f1_score, **kwargs):
        scores = []
        thresholds = []
        best_score = -1
        best_threshold = -1
        for threshold in np.linspace(0, 1, 100):
            prediction_binary = (y_predicted > threshold).astype(int)
            score = metric.__call__(y_true, prediction_binary, **kwargs)
            if score > best_score:
                best_score = score
                best_threshold = threshold
            thresholds.append(threshold)
            scores.append(score)
        return best_threshold
