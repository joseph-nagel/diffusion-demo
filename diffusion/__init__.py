'''
Denoising diffusion.

Summary
-------
A simple but generic implementation of denoising diffusion models is provided.
The goal is merely to establish a starting point for further explorations.

Modules
-------
ddpm : Denoising diffusion model.
layers : Model layers.
schedules : Beta schedules.
unet : U-net architecture.

'''

from . import ddpm
from . import layers
from . import schedules
from . import unet

from .ddpm import DDPM, DDPM2d

from .layers import (
    ConditionalDoubleConv,
    ConditionalResidualBlock,
    ConditionalDense,
    ConditionalDenseModel,
    SinusoidalEncoding,
    LearnableSinusoidalEncoding,
    SelfAttention
)

from .schedules import make_beta_schedule

from .unet import (
    UNet,
    Encoder,
    Decoder,
    Bottleneck
)

