'''DDPM base model.'''

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from ..layers import ClassEmbedding


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
        Optimizer learning rate.

    '''

    def __init__(self,
                 eps_model,
                 betas,
                 criterion='mse',
                 lr=1e-04):

        super().__init__()

        # set trainable epsilon model
        self.eps_model = eps_model

        # set loss function criterion
        if criterion == 'mse':
            self.criterion = nn.MSELoss(reduction='mean')
        elif criterion == 'mae':
            self.criterion = nn.L1Loss(reduction='mean')
        elif callable(criterion):
            self.criterion = criterion
        else:
            raise ValueError('Criterion could not be determined')

        # set initial learning rate
        self.lr = abs(lr)

        # store hyperparams
        self.save_hyperparameters(
            ignore='eps_model',
            logger=True
        )

        # set scheduling parameters
        betas = torch.as_tensor(betas).view(-1) # note that betas[0] corresponds to t = 1.0

        if betas.min() <= 0 or betas.max() >= 1:
            raise ValueError('Invalid beta values encountered')

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        betas_tilde = (1 - alphas_bar[:-1]) / (1 - alphas_bar[1:]) * betas[1:]
        betas_tilde = nn.functional.pad(betas_tilde, pad=(1, 0), value=0.0) # ensure betas_tilde[0] = 0.0

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('betas_tilde', betas_tilde)

    @property
    def num_steps(self):
        '''Get the total number of time steps.'''
        return len(self.betas)

    @property
    def class_cond(self):
        '''Check whether class conditioning is used.'''
        return any([isinstance(m, ClassEmbedding) for m in self.eps_model.modules()])

    @staticmethod
    def _idx2cont_time(tidx, dtype=None):
        '''Transform discrete index to continuous time.'''
        t = tidx + 1 # note that tidx = 0 corresponds to t = 1.0
        t = t.float() if dtype is None else t.to(dtype)
        return t

    def forward(self, x, t, cids=None):
        '''Run the noise-predicting model.'''
        return self.eps_model(x, t, cids=cids)

    def diffuse_step(self, x, tidx):
        '''Simulate single forward process step.'''
        beta = self.betas[tidx]

        eps = torch.randn_like(x)
        x_noisy = (1 - beta).sqrt() * x + beta.sqrt() * eps

        return x_noisy

    def diffuse_all_steps(self, x0):
        '''Simulate and return all forward process steps.'''
        x_noisy = torch.zeros(self.num_steps + 1, *x0.shape, device=x0.device)

        x_noisy[0] = x0
        for tidx in range(self.num_steps):
            x_noisy[tidx + 1] = self.diffuse_step(x_noisy[tidx], tidx)

        return x_noisy

    def diffuse(self, x0, tids, return_eps=False):
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

    def denoise_step(self, x, tids, cids=None, random_sample=False):
        '''Perform single reverse process step.'''

        # set up time variables
        tids = torch.as_tensor(tids, device=x.device).view(-1, 1) # ensure (batch_size>=1, 1)-shaped tensor
        ts = self._idx2cont_time(tids, dtype=x.dtype)

        # set up class labels
        if cids is not None:
            cids = torch.as_tensor(cids, device=x.device).view(-1, 1) # ensure (batch_size>=1, 1)-shaped tensor

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
    def denoise_all_steps(self, xT, cids=None):
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
    def generate(self, sample_shape, cids=None, num_samples=1):
        '''Generate random samples through the reverse process.'''
        x_denoised = torch.randn(num_samples, *sample_shape, device=self.device) # Lightning modules have a device attribute

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

    def loss(self, x, cids=None):
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

    def _get_batch(self, batch):
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

    def training_step(self, batch, batch_idx):
        batch = self._get_batch(batch)

        if isinstance(batch, torch.Tensor):
            loss = self.loss(batch)
        else:
            loss = self.loss(batch[0], cids=batch[1])

        self.log('train_loss', loss.item()) # Lightning logs batch-wise scalars during training per default
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._get_batch(batch)

        if isinstance(batch, torch.Tensor):
            loss = self.loss(batch)
        else:
            loss = self.loss(batch[0], cids=batch[1])

        self.log('val_loss', loss.item()) # Lightning automatically averages scalars over batches for validation
        return loss

    def test_step(self, batch, batch_idx):
        batch = self._get_batch(batch)

        if isinstance(batch, torch.Tensor):
            loss = self.loss(batch)
        else:
            loss = self.loss(batch[0], cids=batch[1])

        self.log('test_loss', loss.item()) # Lightning automatically averages scalars over batches for testing
        return loss

    # TODO: enable LR scheduling
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

