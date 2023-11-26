'''DDPM for 2D data.'''

from .base import DDPM
from .schedules import make_beta_schedule
from ..models import UNet


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
    norm : str
        Normalization type.
    activation : str
        Nonlinearity type.
    embed_dim : int
        Embedding dimension.
    num_resblocks : int
        Number of residual blocks.
    upsample_mode : str
        Convolutional upsampling mode.
    beta_mode : str
        Beta schedule mode.
    beta_range: (float, float)
        Beta range for linear and quadratic schedule.
    cosine_s : float
        Offset for cosine-based schedule.
    sigmoid_range : (float, float)
        Sigmoid range for sigmoid-based schedule.
    num_steps : int
        Number of time steps.
    criterion : {'mse', 'mae'} or callable
        Loss function criterion.
    lr : float
        Optimizer learning rate.

    '''

    def __init__(self,
                 in_channels=1,
                 mid_channels=(16, 32, 64),
                 kernel_size=3,
                 padding=1,
                 norm='batch',
                 activation='leaky_relu',
                 embed_dim=128,
                 num_resblocks=3,
                 upsample_mode='conv_transpose',
                 beta_mode='cosine',
                 beta_range=(1e-04, 0.02),
                 cosine_s=0.008,
                 sigmoid_range=(-5, 5),
                 num_steps=1000,
                 criterion='mse',
                 lr=1e-04):

        # construct U-net model
        eps_model = UNet.from_params(
            in_channels=in_channels,
            mid_channels=mid_channels,
            kernel_size=kernel_size,
            padding=padding,
            norm=norm,
            activation=activation,
            embed_dim=embed_dim,
            num_resblocks=num_resblocks,
            upsample_mode=upsample_mode
        )

        # create noise schedule
        beta_opts = {}
        if beta_mode in ('linear', 'quadratic'):
            beta_opts['beta_range'] = beta_range
        elif beta_mode == 'cosine':
            beta_opts['cosine_s'] = cosine_s
        elif beta_mode == 'sigmoid':
            beta_opts['sigmoid_range'] = sigmoid_range

        betas = make_beta_schedule(num_steps, mode=beta_mode, **beta_opts)

        # initialize DDPM class
        self.save_hyperparameters() # write hyperparams to checkpoints

        super().__init__(
            eps_model=eps_model,
            betas=betas,
            criterion=criterion,
            lr=lr
        )

