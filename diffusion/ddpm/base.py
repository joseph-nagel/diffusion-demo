'''DDPM base model.'''

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from ..layers import ClassEmbedding
from .lr_schedule import make_lr_schedule


# define type aliases
LossType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
BatchType = torch.Tensor | Sequence[torch.Tensor] | dict[str, torch.Tensor]


class DDPM(LightningModule):
    '''
    Plain vanilla DDPM module.

    Summary
    -------
    This module establishes a plain vanilla DDPM variant.
    It is basically a container and wrapper for an
    epsilon model and for the scheduling parameters.
    The class provides methods implementing the forward
    and reverse diffusion processes, respectively.
    Also, the stochastic loss can be computed for training.

    Parameters
    ----------
    eps_model : PyTorch module
        Trainable noise-predicting model.
    betas : array-like
        Beta parameter schedule.
    criterion : {'mse', 'mae'} or callable
        Loss function criterion.
    lr : float
        Initial learning rate.
    lr_schedule : {"constant", "cosine"} or None
        Learning rate schedule type.
    lr_interval : {"epoch", "step"}
        Learning rate update interval.
    lr_warmup : int
        Warmup steps/epochs.

    '''

    def __init__(
        self,
        eps_model: nn.Module,
        betas: torch.Tensor | Sequence[float],
        criterion: str | LossType = 'mse',
        lr: float = 1e-04,
        lr_schedule: str | None = 'constant',
        lr_interval: str = 'epoch',
        lr_warmup: int = 0
    ):

        super().__init__()

        # set trainable epsilon model
        self.set_model(eps_model)

        # set loss function criterion
        if criterion == 'mse':
            self.criterion = nn.MSELoss(reduction='mean')
        elif criterion == 'mae':
            self.criterion = nn.L1Loss(reduction='mean')
        elif callable(criterion):
            self.criterion = criterion
        else:
            raise ValueError('Criterion could not be determined')

        # set LR params
        self.lr = abs(lr)
        self.lr_schedule = lr_schedule
        self.lr_interval = lr_interval
        self.lr_warmup = abs(int(lr_warmup))

        # store hyperparams
        self.save_hyperparameters(
            ignore='eps_model',
            logger=True
        )

        # set noise scheduling params
        betas = torch.as_tensor(betas).view(-1)  # note that betas[0] corresponds to t = 1.0

        if betas.min() <= 0 or betas.max() >= 1:
            raise ValueError('Invalid beta values encountered')

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        betas_tilde = (1 - alphas_bar[:-1]) / (1 - alphas_bar[1:]) * betas[1:]
        betas_tilde = nn.functional.pad(betas_tilde, pad=(1, 0), value=0.0)  # ensure betas_tilde[0] = 0.0

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('betas_tilde', betas_tilde)

    def set_model(self, eps_model: nn.Module) -> None:
        '''Set noise-predicting model.'''
        self.eps_model = eps_model

        # check whether class conditioning is used
        self.class_cond = any([isinstance(m, ClassEmbedding) for m in self.eps_model.modules()])

    @property
    def num_steps(self) -> int:
        '''Get the total number of time steps.'''
        return len(self.betas)

    @staticmethod
    def _idx2cont_time(tidx: torch.Tensor, dtype: str | None = None) -> torch.Tensor:
        '''Transform discrete index to continuous time.'''
        t = tidx + 1  # note that tidx = 0 corresponds to t = 1.0
        t = t.float() if dtype is None else t.to(dtype)
        return t

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cids: torch.Tensor | None = None
    ) -> torch.Tensor:
        '''Run the noise-predicting model.'''
        return self.eps_model(x, t, cids=cids)

    def diffuse_step(self, x: torch.Tensor, tidx: int) -> torch.Tensor:
        '''Simulate single forward process step.'''
        beta = self.betas[tidx]

        eps = torch.randn_like(x)
        x_noisy = (1 - beta).sqrt() * x + beta.sqrt() * eps

        return x_noisy

    def diffuse_all_steps(self, x0: torch.Tensor) -> torch.Tensor:
        '''Simulate and return all forward process steps.'''
        x_noisy = torch.zeros(self.num_steps + 1, *x0.shape, device=x0.device)

        x_noisy[0] = x0
        for tidx in range(self.num_steps):
            x_noisy[tidx + 1] = self.diffuse_step(x_noisy[tidx], tidx)

        return x_noisy

    def diffuse(
        self,
        x0: torch.Tensor,
        tids: torch.Tensor,
        return_eps: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        '''Simulate multiple forward steps at once.'''
        alpha_bar = self.alphas_bar[tids]

        eps = torch.randn_like(x0)

        missing_shape = [1] * (eps.ndim - alpha_bar.ndim)
        alpha_bar = alpha_bar.view(*alpha_bar.shape, *missing_shape)

        x_noisy = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps

        if return_eps:
            return x_noisy, eps
        else:
            return x_noisy

    def denoise_step(
        self,
        x: torch.Tensor,
        tids: torch.Tensor,
        cids: torch.Tensor | None = None,
        random_sample: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        '''Perform single reverse process step.'''

        # set up time variables
        tids = torch.as_tensor(tids, device=x.device).view(-1, 1)  # ensure (batch_size>=1, 1)-shaped tensor
        ts = self._idx2cont_time(tids, dtype=x.dtype)

        # set up class labels
        if cids is not None:
            cids = torch.as_tensor(cids, device=x.device).view(-1, 1)  # ensure (batch_size>=1, 1)-shaped tensor

        # predict eps based on noisy x and t
        eps_pred = self.eps_model(x, ts, cids=cids)

        # compute mean
        p = 1 / self.alphas[tids].sqrt()
        q = self.betas[tids] / (1 - self.alphas_bar[tids]).sqrt()

        missing_shape = [1] * (eps_pred.ndim - ts.ndim)
        p = p.view(*p.shape, *missing_shape)
        q = q.view(*q.shape, *missing_shape)

        x_denoised_mean = p * (x - q * eps_pred)

        # retrieve variance
        x_denoised_var = self.betas_tilde[tids]
        # x_denoised_var = self.betas[tids]

        # generate random sample
        if random_sample:
            eps = torch.randn_like(x_denoised_mean)
            x_denoised = x_denoised_mean + x_denoised_var.sqrt() * eps

        if random_sample:
            return x_denoised
        else:
            return x_denoised_mean, x_denoised_var

    @torch.no_grad()
    def denoise_all_steps(self, xT: torch.Tensor, cids: torch.Tensor | None = None) -> torch.Tensor:
        '''Perform and return all reverse process steps.'''
        x_denoised = torch.zeros(self.num_steps + 1, *(xT.shape), device=xT.device)

        x_denoised[0] = xT
        for idx, tidx in enumerate(reversed(range(self.num_steps))):

            # generate random sample
            if tidx > 0:
                x_denoised[idx + 1] = self.denoise_step(
                    x_denoised[idx],
                    tidx,
                    cids=cids,
                    random_sample=True
                )

            # take the mean in the last step
            else:
                x_denoised[idx + 1], _ = self.denoise_step(
                    x_denoised[idx],
                    tidx,
                    cids=cids,
                    random_sample=False
                )

        return x_denoised

    @torch.no_grad()
    def generate(
        self,
        sample_shape: Sequence[int],
        cids: torch.Tensor | None = None,
        num_samples: int = 1
    ) -> torch.Tensor:
        '''Generate random samples through the reverse process.'''
        x_denoised = torch.randn(num_samples, *sample_shape, device=self.device)  # Lightning modules have a device attribute

        for tidx in reversed(range(self.num_steps)):

            # generate random sample
            if tidx > 0:
                x_denoised = self.denoise_step(
                    x_denoised,
                    tidx,
                    cids=cids,
                    random_sample=True
                )

            # take the mean in the last step
            else:
                x_denoised, _ = self.denoise_step(
                    x_denoised,
                    tidx,
                    cids=cids,
                    random_sample=False
                )

        return x_denoised

    def loss(self, x: torch.Tensor, cids: torch.Tensor | None = None) -> torch.Tensor:
        '''Compute stochastic loss.'''

        # draw random time steps
        tids = torch.randint(0, self.num_steps, size=(x.shape[0], 1), device=x.device)
        ts = self._idx2cont_time(tids, dtype=x.dtype)

        # perform forward process steps
        x_noisy, eps = self.diffuse(x, tids, return_eps=True)

        # predict eps based on noisy x and t
        eps_pred = self.eps_model(x_noisy, ts, cids=cids)

        # compute loss
        loss = self.criterion(eps_pred, eps)

        return loss

    def _get_batch(self, batch: BatchType) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        '''Get batch features and labels (if needed).'''

        if isinstance(batch, torch.Tensor):
            x_batch = batch

            if self.class_cond:
                raise RuntimeError('No labels found')

        elif isinstance(batch, (tuple, list)):
            x_batch = batch[0]

            if self.class_cond:
                y_batch = batch[1]

        elif isinstance(batch, dict):
            x_batch = batch['features']

            if self.class_cond:
                y_batch = batch['labels']

        else:
            raise TypeError('Invalid batch type encountered: {}'.format(type(batch)))

        if self.class_cond:
            return x_batch, y_batch
        else:
            return x_batch

    def training_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        batch = self._get_batch(batch)

        if isinstance(batch, torch.Tensor):
            loss = self.loss(batch)
        else:
            loss = self.loss(batch[0], cids=batch[1])

        self.log('train_loss', loss.item())  # Lightning logs batch-wise scalars during training per default
        return loss

    def validation_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        batch = self._get_batch(batch)

        if isinstance(batch, torch.Tensor):
            loss = self.loss(batch)
        else:
            loss = self.loss(batch[0], cids=batch[1])

        self.log('val_loss', loss.item())  # Lightning automatically averages scalars over batches for validation
        return loss

    def test_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        batch = self._get_batch(batch)

        if isinstance(batch, torch.Tensor):
            loss = self.loss(batch)
        else:
            loss = self.loss(batch[0], cids=batch[1])

        self.log('test_loss', loss.item())  # Lightning automatically averages scalars over batches for testing
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer | tuple[list, list]:

        # create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # return optimizer only (if no LR schedule has been set)
        if self.lr_schedule is None:
            return optimizer

        # create LR schedule (if a schedule has been set)
        else:

            # get total number of training time units
            if self.lr_interval == 'epoch':
                num_total = self.trainer.max_epochs
            elif self.lr_interval == 'step':
                num_total = self.trainer.estimated_stepping_batches
            else:
                raise ValueError(f'Unknown LR interval: {self.lr_interval}')

            # create LR scheduler
            lr_scheduler = make_lr_schedule(
                optimizer=optimizer,
                mode=self.lr_schedule,
                num_total=num_total,
                num_warmup=self.lr_warmup,
                last_epoch=-1
            )

            # create LR config
            lr_config = {
                'scheduler': lr_scheduler,  # set LR scheduler
                'interval': self.lr_interval,  # set time unit (step or epoch)
                'frequency': 1  # set update frequency
            }

            return [optimizer], [lr_config]

