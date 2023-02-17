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

from src.image_transforms import crop_image, random_sharpness_or_blur


def crop_black_and_white_loader(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return crop_image(img)


class CTDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.test_shufle = test_shufle

        self.train_transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=.2, contrast=.7, saturation=.1, hue=.1),
            random_sharpness_or_blur,
        ])

        self.base_transform = transforms.Compose([
            transforms.Resize(
                (128, 98), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.num_classes = 2

    @property
    def n_images(self):
        class_dirs = os.listdir(self.data_dir)
        n_files_1 = len(os.listdir(os.path.join(self.data_dir, class_dirs[0])))
        n_files_2 = len(os.listdir(os.path.join(self.data_dir, class_dirs[1])))
        return n_files_1 + n_files_2

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.dataset = datasets.ImageFolder(self.data_dir,
                                                loader=crop_black_and_white_loader,
                                                transform=transforms.transforms.Compose([self.base_transform,
                                                                                         self.train_transform])
                                                )

            self.data_train, self.data_validation = random_split(self.dataset,
                                                                 [round(len(self.dataset.samples) * 0.8),
                                                                  round(len(self.dataset.samples) * 0.2)])

        if stage == 'predict':
            self.dataset = NoLabelDataset(self.data_dir,
                                          transform=self.base_transform)

        if stage == 'test':
            self.dataset = datasets.ImageFolder(self.data_dir,
                                                loader=crop_black_and_white_loader,
                                                transform=self.base_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data_train,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data_validation,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=True
                                           )


class NoLabelDataset(VisionDataset):
    """Used for folders without labels."""

    def __getitem__(self, index):
        image_files = os.listdir(self.root)
        path_image = os.path.join(self.root, image_files[index])
        sample = crop_black_and_white_loader(path_image)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(os.listdir(self.root))
