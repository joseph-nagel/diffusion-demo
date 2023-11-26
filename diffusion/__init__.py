'''
Denoising diffusion.

Summary
-------
A simple but generic implementation of denoising diffusion models is provided.
The goal is merely to establish a starting point for further explorations.

Modules
-------
ddpm : Denoising diffusion models.
layers : Model layers.
models : Prediction models.

'''

from . import ddpm
from . import layers
from . import models


from .ddpm import (
    DDPM,
    DDPM2d,
    make_beta_schedule
)

from .layers import (
    SelfAttention,
    DoubleConv,
    CondDoubleConv,
    ResidualBlock,
    CondResidualBlock,
    CondDense,
    SinusoidalEncoding,
    LearnableSinusoidalEncoding,
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

