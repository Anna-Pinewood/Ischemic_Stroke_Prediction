import numpy as np
import torch
import torch_ard as nn_ard
from pytorch_lightning.core.module import LightningModule
import torch.nn as nn
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_ard import get_ard_reg

from src.CTDataModule import IMG_HEIGHT, IMG_WIDTH
from src.utils import maxpool_output_shape


class ConvBlockArd(nn.Module):
    def __init__(self, in_channels, out_chanels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn_ard.Conv2dARD(
            in_channels, out_chanels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_chanels)

    def forward(self, x):
        return nn.functional.relu(self.bn(self.conv(x)))


class InceptionBlockArd(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_1x1: int,
                 reduce_3x3: int,
                 out_3x3: int,
                 reduce_5x5: int,
                 out_5x5: int,
                 out_pooling: int):
        super().__init__()
        self.branch_1filter = ConvBlockArd(in_channels, out_1x1, kernel_size=1)

        self.branch_3filter = nn.Sequential(ConvBlockArd(in_channels, reduce_3x3, kernel_size=1),
                                            ConvBlockArd(
                                                reduce_3x3, out_3x3, kernel_size=3, padding=1)
                                            )
        self.branch_5filter = nn.Sequential(
            ConvBlockArd(in_channels, reduce_5x5, kernel_size=1),
            ConvBlockArd(reduce_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch_maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlockArd(in_channels, out_pooling, kernel_size=1))

    def forward(self, x):
        branches = (self.branch_1filter, self.branch_3filter,
                    self.branch_5filter, self.branch_maxpool)
        output = torch.cat([branch(x) for branch in branches], 1)
        return output


class SiameseDiffBlockArd(nn.Module):

    def __init__(self):
        super().__init__()

        self.inception_chain = nn.Sequential(
            InceptionBlockArd(in_channels=1,
                              out_1x1=64,
                              reduce_3x3=64,
                              out_3x3=64,
                              reduce_5x5=64,
                              out_5x5=64,
                              out_pooling=64),
            InceptionBlockArd(256, 64, 64, 64, 64, 64, 64),
            InceptionBlockArd(256, 64, 64, 64, 64, 64, 64),
            InceptionBlockArd(256, 64, 64, 64, 64, 64, 64)
        )

    def forward(self, x):
        input1 = self._split_tensor(x, width=IMG_WIDTH)
        input2 = self._split_tensor(x, left=False, width=IMG_WIDTH)

        output = self.inception_chain(torch.cat([input1, input2], dim=0))
        N = x.shape[0]
        return abs(output[0:N] - torch.flip(output[N:], dims=[3]))

    @staticmethod
    def _split_tensor(
            batch: torch.Tensor,
            left: bool = True,
            width: int = IMG_WIDTH):
        return batch[:, :, :, :(width // 2)] if left \
            else batch[:, :, :, (width // 2):]


class DeepSymNetArd(LightningModule):
    def __init__(self):
        super().__init__()
        self.image_height, self.image_width = IMG_HEIGHT, IMG_WIDTH
        self.siamese_part = SiameseDiffBlockArd()
        self.shared_tunnel = nn.Sequential(
            InceptionBlockArd(256, 64, 64, 64, 64, 64, 64),
            InceptionBlockArd(256, 64, 64, 64, 64, 64, 64),
            nn.MaxPool2d(kernel_size=3, padding=1))
        height_new, width_new = maxpool_output_shape(
            input_size=(IMG_HEIGHT, IMG_WIDTH // 2),
            layer=self.shared_tunnel[-1])
        units_fc = height_new * width_new * 256
        self.fc1 = nn_ard.LinearARD(units_fc, 1)

        self.learning_rate = 1e-5

    def forward(self, x):
        output_siam = self.siamese_part(x)
        output = self.shared_tunnel(output_siam)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        output = nn.Sigmoid()(output)
        output = output.reshape(output.shape[0])
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode='min', patience=12, verbose=True),
                "monitor": "val_loss",
                "frequency": 1},
        }

    def loss(self,
             y_predicted,
             y_true,
             loss_weight=1.,
             kl_weight=0.00001):
        y_true = y_true.to(torch.float32)
        loss_fn = nn.BCELoss()(y_predicted, y_true)
        return loss_weight * loss_fn \
            + kl_weight * get_ard_reg(self)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("training_loss", loss, on_epoch=True,
                 on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        if len(torch.unique(y)) < 2:

            roc_auc = -1
        else:
            roc_auc = roc_auc_score(y, y_hat.detach().numpy())
        rmse = np.sqrt(mean_squared_error(y, y_hat.detach().numpy()))
        self.log('roc_auc', roc_auc)
        self.log('rmse', rmse)
        return {"roc_auc":  roc_auc, "rmse": rmse,
                "y_true": y, "y_pred": y_hat}
