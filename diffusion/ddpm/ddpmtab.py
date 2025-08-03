'''DDPM for tabular data.'''

from ..models import CondDenseModel
from .base import DDPM
from .noise_schedule import make_beta_schedule


class DDPMTab(DDPM):
    '''
    DDPM for problems with tabular data.

    Summary
    -------
    A DDPM for tabular data structures is implemented.
    It uses a fully connected model for predicting the noise.

    Parameters
    ----------
    in_features : int
        Number of input features.
    mid_features : list or tuple of ints
        Hidden layer feature numbers.
    activation : str
        Nonlinearity type.
    embed_dim : int
        Dimension of the time embedding.
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
        in_features=2,
        mid_features=(128, 128, 128),
        activation='leaky_relu',
        embed_dim=128,
        num_steps=500,
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

        # construct dense model
        num_features = (in_features, *mid_features, in_features)

        eps_model = CondDenseModel(
            num_features=num_features,
            activation=activation,
            embed_dim=embed_dim
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

