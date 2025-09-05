'''Swiss roll datamodule.'''

import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningDataModule


def make_swiss_roll_2d(
    num_samples: int,
    noise_level: float = 0.5,
    scaling: float = 0.15,
    random_state: int | None = None,
    test_size: float | int | None = None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    '''
    Create 2D Swiss roll data.

    Parameters
    ----------
    num_samples : int
        Number of samples to create.
    noise_level : float
        Noise standard deviation.
    scaling : float
        Scaling parameter.
    random_state : int or None
        Random generator seed.
    test_size : float, int or None
        Test size parameter.

    '''

    # create 3D data
    x, _ = make_swiss_roll(
        num_samples,
        noise=abs(noise_level),
        random_state=random_state
    )

    # restrict to 2D
    x = x[:, [0, 2]]

    # scale data
    x = scaling * x

    # return
    if test_size is None:
        return x

    # split data and return
    else:
        x_train, x_val = train_test_split(
            x,
            test_size=test_size
        )

        return x_train, x_val


class SwissRollDataModule(LightningDataModule):
    '''
    DataModule for 2D Swiss roll data.

    Parameters
    ----------
    num_train : int
        Number of training samples.
    num_val : int
        Number of validation samples.
    num_test : int
        Number of testing samples.
    noise_level : float
        Noise standard deviation.
    scaling : float
        Scaling parameter.
    random_state : int
        Random generator seed.
    batch_size : int
        Batch size of the data loader.
    num_workers : int
        Number of workers for the loader.

    '''

    def __init__(
        self,
        num_train: int,
        num_val: int = 0,
        num_test: int = 0,
        noise_level: float = 0.5,
        scaling: float = 0.15,
        random_state: int = 42,
        batch_size: int = 32,
        num_workers: int = 0
    ):

        super().__init__()

        # set data parameters
        self.num_train = abs(int(num_train))
        self.num_val = abs(int(num_val))
        self.num_test = abs(int(num_test))
        self.noise_level = abs(noise_level)
        self.scaling = scaling

        # set random state
        self.random_state = random_state

        # set loader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        '''Prepare numerical data.'''

        # create data
        num_samples = self.num_train + self.num_val + self.num_test

        x = make_swiss_roll_2d(
            num_samples,
            noise_level=self.noise_level,
            scaling=self.scaling,
            random_state=self.random_state,
            test_size=None
        )

        # transform to tensor
        self.x = torch.tensor(x, dtype=torch.float32)

    @property
    def x_train(self) -> torch.Tensor:
        return self.x[:self.num_train]

    @property
    def x_val(self) -> torch.Tensor:
        return self.x[self.num_train:self.num_train+self.num_val]

    @property
    def x_test(self) -> torch.Tensor:
        return self.x[self.num_train+self.num_val:]

    def setup(self, stage: str):
        '''Set up train/test/val. datasets.'''

        # create train/val. datasets
        if stage in ('fit', 'validate'):
            self.train_set = TensorDataset(self.x_train)
            self.val_set = TensorDataset(self.x_val)

        # create test dataset
        elif stage == 'test':
            self.test_set = TensorDataset(self.x_test)

    def train_dataloader(self) -> DataLoader:
        '''Create train dataloader.'''
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        '''Create val. dataloader.'''
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

    def test_dataloader(self) -> DataLoader:
        '''Create test dataloader.'''
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )
