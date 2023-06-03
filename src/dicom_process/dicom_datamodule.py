import logging
import os
import SimpleITK as sitk
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import pydicom
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import random_split, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

from src.image_transforms import crop_image, random_sharpness_or_blur

IMG_HEIGHT = 500
IMG_WIDTH = 500

logger = logging.getLogger(__name__)


class DicomDataModule(pl.LightningDataModule):  # pylint: disable=too-many-instance-attributes
    """Treat CT brain scans."""

    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 throw_out_random: float = 0.,
                 test_shuffle: bool = True,
                 img_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.throw_out_random = throw_out_random
        self.test_shuffle = test_shuffle

        self.image_height, self.image_width = IMG_HEIGHT, IMG_WIDTH
        if img_size is not None:
            self.image_height, self.image_width = img_size

        self.train_transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=.2, contrast=.7, saturation=.1, hue=.1),
            random_sharpness_or_blur,
        ])

        self.base_transform = transforms.Compose([
            transforms.Resize(
                (IMG_HEIGHT, IMG_WIDTH), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.num_classes = 2

        self.dataset = None
        self.data_train:  Optional[datasets.ImageFolder] = None
        self.data_validation: Optional[datasets.ImageFolder] = None

    @property
    def n_images(self) -> int:
        """How many initial images are there in datamodule."""
        class_dirs = [path for path in Path(self.data_dir).glob("*") if path.is_dir()]
        n_files_1 = len(os.listdir(os.path.join(self.data_dir, class_dirs[0])))
        n_files_2 = len(os.listdir(os.path.join(self.data_dir, class_dirs[1])))
        return n_files_1 + n_files_2

    @property
    def n_stay_images(self) -> int:
        """How many images are stayed in datamodule."""
        n_stay_files = round(
            self.n_images - self.throw_out_random * self.n_images)
        return n_stay_files

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:

            self.dataset = DicomDataset(self.data_dir)

            self.dataset = torch.utils.data.Subset(self.dataset,
                                                   np.random.choice(len(self.dataset),
                                                                    self.n_stay_images, replace=True)
                                                   )

            self.data_train, self.data_validation = random_split(self.dataset,
                                                                 [round(len(self.dataset) * 0.8),
                                                                  round(len(self.dataset) * 0.2)])

            logger.info('Num train images: %s', str(len(self.data_train)))
            logger.info('Num valid images: %s', str(len(self.data_validation)))

        if stage == 'test':
            self.dataset = DicomDataset(self.data_dir)

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
                                           shuffle=self.test_shuffle
                                           )


class DicomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        sub_pathes = [path for path in self.root_dir.glob("*") if path.is_dir()]

        patients_one = [path for path in sub_pathes[0].glob("*") if path.is_dir()]
        patients_two = [path for path in sub_pathes[1].glob("*") if path.is_dir()]
        self.patients = patients_one + patients_two
        self.labels = [0] * len(patients_one) + [1] * len(patients_two)

        self.patient_tensors = [self.read_process_sample(patient) for patient in self.patients]

    def read_patient(self, path):
        # dcm_files = list(path.glob("*.dcm"))
        dcm_files = list(path.glob("*"))
        tensor_list = []
        for dcm in tqdm(dcm_files):
            dcm_read = pydicom.dcmread(dcm)
            dcm_img = dcm_read.pixel_array.astype(float)
            dcm_img_norm = (dcm_img - np.min(dcm_img)) / \
                (np.max(dcm_img) - np.min(dcm_img))
            dcm_tensor = torch.Tensor(dcm_img_norm)
            tensor_list.append(dcm_tensor)
        stacked_dcm = torch.stack(tensor_list)
        stacked_dcm_dim = stacked_dcm.unsqueeze(0)
        return stacked_dcm_dim


    def cut_zeros(self, tensor: torch.Tensor):
        # Find the indices of elements that are greater than the 10th percentile of tensor values
        # Get the 10th percentile value
        percentile_value = np.percentile(tensor.numpy(), 10)

        # Find the indices of values greater than or equal to the percentile value along each dimension
        nonzero_depth = torch.unique(
            torch.nonzero(torch.sum(torch.sum(tensor >= percentile_value, dim=2), dim=2)).squeeze())
        nonzero_rows = torch.unique(torch.nonzero(
            torch.sum(torch.sum(tensor[:, nonzero_depth, :, :] >= percentile_value, dim=1), dim=1)).squeeze())
        nonzero_cols = torch.unique(torch.nonzero(
            torch.sum(torch.sum(tensor[:, nonzero_depth, :, :] >= percentile_value, dim=1), dim=0)).squeeze())

        # Remove rows and columns with only zeros or values less than the percentile value
        tensor = tensor[:, nonzero_depth, :, :]
        tensor = tensor[:, :, nonzero_rows, :]
        tensor = tensor[:, :, :, nonzero_cols]
        return tensor

    import SimpleITK as sitk
    import torch

    def resample_tensor(self, tensor, spacing, size):
        """
        Resample a PyTorch tensor using the SimpleITK library.

        Args:
            tensor: PyTorch tensor of shape [1, D, H, W]
            spacing: Tuple of new voxel spacing in z, y, and x directions
            size: Tuple of new image size in D, H, and W dimensions

        Returns:
            Resampled PyTorch tensor of shape [1, D_new, H_new, W_new]
        """
        # Convert PyTorch tensor to SimpleITK image
        image = sitk.GetImageFromArray(tensor[0].numpy())
        image.SetSpacing(spacing)

        # Create resampled image with the desired size
        resampled = sitk.Resample(image, list(size), sitk.Transform(), sitk.sitkLinear, image.GetOrigin(),
                                  (spacing[0] * size[0], spacing[1] * size[1], spacing[2] * size[2]),
                                  image.GetDirection(), 0.0,
                                  image.GetPixelID())

        # Convert SimpleITK image to PyTorch tensor
        resampled_tensor = torch.from_numpy(sitk.GetArrayFromImage(resampled)).unsqueeze(0)

        return resampled_tensor

    def read_process_sample(self, patient_path):
        patient_tensor = self.read_patient(patient_path)
        patient_valid = self.cut_zeros(patient_tensor)
        patient_resampled = self.resample_tensor(patient_valid, spacing=(0.5, 0.5, 0.5),size = (128, 256, 256))
        return patient_valid

    def __getitem__(self, idx):
        label = self.labels[idx]
        patient_tensor = self.patient_tensors[idx]
        return patient_tensor, label

    def __len__(self):
        return len(self.patients)
