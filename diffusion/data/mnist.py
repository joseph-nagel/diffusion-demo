'''MNIST datamodule.'''

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from lightning import LightningDataModule


class MNISTDataModule(LightningDataModule):
    '''
    DataModule for MNIST-like datasets.

    Parameters
    ----------
    data_set : str
        Determines the MNIST-like dataset.
    data_dir : str
        Directory for storing the data.
    mean : float
        Mean for data normalization.
    std : float
        Standard deviation for normalization.
    random_state : int
        Random generator seed.
    batch_size : int
        Batch size of the data loader.
    num_workers : int
        Number of workers for the loader.

    '''

    def __init__(self,
                 data_set='mnist',
                 data_dir='.',
                 mean=None,
                 std=None,
                 random_state=42,
                 batch_size=32,
                 num_workers=0):

        super().__init__()

        # set dataset
        if data_set == 'mnist':
            self.data_class = datasets.MNIST
        elif data_set in ('fashion_mnist', 'fmnist'):
            self.data_class = datasets.FashionMNIST
        elif data_set in ('kuzushiji_mnist', 'kmnist'):
            self.data_class = datasets.KMNIST
        else:
            raise ValueError(f'Invalid dataset: {data_set}')

        # set data location
        self.data_dir = data_dir

        # set random state
        self.random_state = random_state

        # set loader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

        # create transforms
        train_transforms = [
            transforms.RandomRotation(5), # TODO: refine data augmentation
            transforms.ToTensor()
        ]

        test_transforms = [transforms.ToTensor()]

        if (mean is not None) and (std is not None):
            normalize_fn = transforms.Normalize(mean=mean, std=std)

            train_transforms.append(normalize_fn)
            test_transforms.append(normalize_fn)

        self.train_transform = transforms.Compose(train_transforms)
        self.test_transform = transforms.Compose(test_transforms)

    def prepare_data(self):
        '''Download data.'''

        train_set = self.data_class(
            self.data_dir,
            train=True,
            download=True
        )

        test_set = self.data_class(
            self.data_dir,
            train=False,
            download=True
        )

    def setup(self, stage):
        '''Set up train/test/val. datasets.'''

        # create train/val. datasets
        if stage in ('fit', 'validate'):
            train_set = self.data_class(
                self.data_dir,
                train=True,
                transform=self.train_transform
            )

            self.train_set, self.val_set = random_split(
                train_set,
                [50000, 10000],
                generator=torch.Generator().manual_seed(self.random_state)
            )

        # create test dataset
        elif stage == 'test':
            self.test_set = self.data_class(
                self.data_dir,
                train=False,
                transform=self.test_transform
            )

    def train_dataloader(self):
        '''Create train dataloader.'''
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

    def val_dataloader(self):
        '''Create val. dataloader.'''
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

    def test_dataloader(self):
        '''Create test dataloader.'''
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

