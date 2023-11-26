'''Some utilities.'''

from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import train_test_split
import torch


def make_swissroll_2d(num_samples,
                      noise_level=0.5,
                      scaling=0.1,
                      val_size=0.2,
                      dtype='float32',
                      as_tensor=True):
    '''Create 2D swiss roll data.'''

    # create 3D data
    x, _ = make_swiss_roll(num_samples, noise=noise_level)

    # restrict to 2D
    x = x[:, [0, 2]]

    # scale
    x = scaling * x

    # split into train and val. set
    if val_size is not None:
        x_train, x_val = train_test_split(x, test_size=val_size)
    else:
        x_train = x
        x_val = None

    # set float precision
    x_train = x_train.astype(dtype)

    if x_val is not None:
        x_val = x_val.astype(dtype)

    # convert to tensors
    if as_tensor:
        x_train = torch.as_tensor(x_train)

        if x_val is not None:
            x_val = torch.as_tensor(x_val)

    if x_val is not None:
        return x_train, x_val
    else:
        return x_train

