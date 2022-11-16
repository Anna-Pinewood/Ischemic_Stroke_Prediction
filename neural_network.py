import logging

import torch
import torch.nn as nn
from pytorch_lightning.core.module import LightningModule

from torch.utils.tensorboard import SummaryWriter
tb = SummaryWriter()


LOGGER = logging.getLogger()
# LOGGER.setLevel(logging.DEBUG)
# LOGGER.debug('Test Message')

def split_tensor(batch, left=True, length_image=98):
  """
  batch is output of next(iter(dm_loader))[0]
  of shape   torch.Size([ n_samples_in_batch, 1, 128, 98])
  Returns batch of left parts or batch of right.
  """
  return batch[:, :,:,:(length_image//2) ] if left else batch[:, :,:, (length_image//2): ]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_chanels)

    def forward(self, x):
        return nn.functional.relu(self.bn(self.conv(x)))
        # return nn.functional.relu(self.conv(x))


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
                                            ConvBlock(reduce_3x3, out_3x3, kernel_size=3, padding=1)
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
        branches = (self.branch_1filter, self.branch_3filter, self.branch_5filter, self.branch_maxpool)
        output = torch.cat([branch(x) for branch in branches], 1)
        return output
#create the Siamese Neural Network

class SiameseAndDifferenceBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.inception_chain = nn.Sequential(
            InceptionBlock(in_channels=1, out_1x1=64, reduce_3x3=64, out_3x3=64, reduce_5x5=64, out_5x5=64, out_pooling=64),
            InceptionBlock(256, 64, 64, 64, 64, 64, 64),
            InceptionBlock(256, 64, 64, 64, 64, 64, 64),
            InceptionBlock(256, 64, 64, 64, 64, 64, 64)
        )

    def forward(self, x):
        input1 = split_tensor(x, length_image=128)
        input2 = split_tensor(x, left=False, length_image=128)
        output1 = self.inception_chain(input1)
        output2 = self.inception_chain(input2)

        return abs(output1 - output2) # !!! absolute


class DeepSymNet(LightningModule):
    def __init__(self):
        super().__init__()

        self.siamese_part = SiameseAndDifferenceBlock()

        self.shared_tunnel = nn.Sequential(
            InceptionBlock(256, 64, 64, 64, 64, 64, 64),
            InceptionBlock(256, 64, 64, 64, 64, 64, 64),
            nn.MaxPool2d(kernel_size=3, padding=1)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(185856, 1)  # !!!!!
        )

    def forward(self, x):
        LOGGER.debug(f'Input shape is {x.shape}')
        output_siam = self.siamese_part(x)
        LOGGER.debug(f'Output of SiameseAndDifferenceBlock block is {output_siam.shape}')
        output = self.shared_tunnel(output_siam)
        LOGGER.debug(f'Output of IM-IM-MP block is {output.shape}')

        output = output.view(output_siam.size()[0], -1)
        LOGGER.debug(f'Reshaping for fully connected to {output.shape}')
        output = self.fc1(output)
        output = nn.Sigmoid()(output)
        LOGGER.debug(f'Output shape is {output.shape}')
        output = output.reshape(output.shape[0])

        LOGGER.debug(f'Output reshaped to {output.shape}')
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def binary_cross_entropy_loss(self, y_predicted, y_true):
        y_true = y_true.to(torch.float32)
        LOGGER.debug(f'y_pred \n {y_predicted}')
        LOGGER.debug(f'y_true \n {y_true}')
        loss = nn.BCELoss()(y_predicted, y_true)
        return loss

    def training_step(self, batch, batch_idx):
        # LOGGER.debug(f'Batch \n {batch[1]}')
        x, y = batch
        LOGGER.debug(f'Model \n  {self.forward(x)}')
        y_hat = self.forward(x)
        loss = self.binary_cross_entropy_loss(y_hat, y)
        self.log("training_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = self.binary_cross_entropy_loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
