'''DDPM for 2D data.'''

from ..models import UNet
from .base import DDPM
from .noise_schedule import make_beta_schedule


class DDPM2d(DDPM):
    '''
    DDPM for problems with two spatial dimensions.

    Summary
    -------
    This subclass facilitates the construction of a 2D DDPM.
    It employs a U-net-based noise model and a beta schedule.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : list or tuple of ints
        Hidden layer channel numbers.
    kernel_size : int
        Convolutional kernel size.
    padding : int
        Padding parameter.
    norm : str or None
        Normalization type.
    activation : str or None
        Nonlinearity type.
    num_resblocks : int
        Number of residual blocks.
    upsample_mode : str
        Convolutional upsampling mode.
    embed_dim : int
        Dimension of the time embedding.
    num_classes : int or None
        Number of classes (for conditioning).
    num_steps : int
        Number of time steps.
    schedule : str
        Determines the noise scheduling type.
    beta_range: (float, float)
        Beta range for linear and quadratic schedules.
    cosine_s : float
        Offset parameter for cosine-based alpha_bar.
    sigmoid_range : (float, float)
        Input range for evaluating the sigmoid in
        the corresponding sqrt.(alpha_bar) schedule.
    criterion : {'mse', 'mae'} or callable
        Loss function criterion.
    lr : float
        Initial learning rate.
    lr_schedule : {"constant", "cosine"}
        Learning rate schedule type.
    lr_interval : {"epoch", "step"}
        Learning rate update interval.
    lr_warmup : int
        Warmup steps/epochs.

    '''

    def __init__(
        self,
        in_channels=1,
        mid_channels=(16, 32, 64),
        kernel_size=3,
        padding=1,
        norm='batch',
        activation='leaky_relu',
        num_resblocks=3,
        upsample_mode='conv_transpose',
        embed_dim=128,
        num_classes=None,
        num_steps=1000,
        schedule='cosine',
        beta_range=(1e-04, 0.02),
        cosine_s=0.008,
        sigmoid_range=(-5, 5),
        criterion='mse',
        lr=1e-04,
        lr_schedule='constant',
        lr_interval='epoch',
        lr_warmup=0
    ):

        # construct U-net model
        eps_model = UNet.from_params(
            in_channels=in_channels,
            mid_channels=mid_channels,
            kernel_size=kernel_size,
            padding=padding,
            norm=norm,
            activation=activation,
            num_resblocks=num_resblocks,
            upsample_mode=upsample_mode,
            embed_dim=embed_dim,
            num_classes=num_classes
        )

        # create noise schedule
        betas = make_beta_schedule(
            num_steps,
            mode=schedule,
            beta_range=beta_range,
            cosine_s=cosine_s,
            sigmoid_range=sigmoid_range
        )

        # initialize DDPM class
        super().__init__(
            eps_model=eps_model,
            betas=betas,
            criterion=criterion,
            lr=lr,
            lr_schedule=lr_schedule,
            lr_interval=lr_interval,
            lr_warmup=lr_warmup
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)

