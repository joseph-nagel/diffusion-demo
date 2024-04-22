'''Swiss-roll datamodule.'''

from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningDataModule


def make_swiss_roll_2d(num_samples,
                       noise_level=0.5,
                       scaling=0.15,
                       random_state=None,
                       test_size=None):
    '''
    Create 2D swiss-roll data.

    Parameters
    ----------
    num_samples : int
        Number of samples to create.
    noise_level : float
        Noise standard deviation.
    scaling : float
        Scaling parameter.
    random_state : int
        Random generator seed.
    test_size : int or float
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
    DataModule for 2D swiss-roll data.

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

    def __init__(self,
                 num_train,
                 num_val=0,
                 num_test=0,
                 noise_level=0.5,
                 scaling=0.15,
                 random_state=42,
                 batch_size=32,
                 num_workers=0):

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
    def x_train(self):
        return self.x[:self.num_train]

    @property
    def x_val(self):
        return self.x[self.num_train:self.num_train+self.num_val]

    @property
    def x_test(self):
        return self.x[self.num_train+self.num_val:]

    def setup(self, stage):
        '''Set up train/test/val. datasets.'''

        # create train/val. datasets
        if stage in ('fit', 'validate'):
            self.train_set = TensorDataset(self.x_train)
            self.val_set = TensorDataset(self.x_val)

        # create test dataset
        elif stage == 'test':
            self.test_set = TensorDataset(self.x_test)

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

