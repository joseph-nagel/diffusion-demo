'''
Denoising diffusion.

Summary
-------
A simple but generic implementation of denoising diffusion models is provided.
The goal is merely to establish a starting point for further explorations.

Modules
-------
data : Datamodules.
ddpm : Denoising diffusion models.
layers : Model layers.
models : Prediction models.

'''

from . import (
    data,
    ddpm,
    layers,
    models
)

from .data import (
    make_swiss_roll_2d,
    SwissRollDataModule,
    MNISTDataModule
)

from .ddpm import (
    DDPM,
    DDPM2d,
    DDPMTab,
    make_beta_schedule
)

from .layers import (
    SelfAttention2D,
    DoubleConv,
    CondDoubleConv,
    ResidualBlock,
    CondResidualBlock,
    CondDense,
    SinusoidalEncoding,
    LearnableSinusoidalEncoding,
    ClassEmbedding,
    make_dense,
    make_conv,
    make_activation,
    make_norm
)

from .models import (
    CondDenseModel,
    UNet,
    UNetEncoder,
    UNetDecoder,
    UNetBottleneck
)

