'''
Datamodules.

Modules
-------
mnist : MNIST datamodule.
swissroll : Swiss roll datamodule.

'''

from . import mnist, swissroll

from .mnist import MNISTDataModule

from .swissroll import make_swiss_roll_2d, SwissRollDataModule
