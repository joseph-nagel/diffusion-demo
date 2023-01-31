'''Denoising diffusion model.'''

import torch
import torch.nn as nn
import pytorch_lightning as pl

class DDPM(pl.LightningModule):
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

    '''

    def __init__(self,
                 eps_model,
                 betas,
                 criterion='mse'):
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

    def diffuse_step(self, x, tidx):
        '''Simulate single forward process step.'''
        beta = self.betas[tidx] # note that tidx = 0 corresponds to t = 1.0
        eps = torch.randn_like(x)
        x_noisy = (1 - beta).sqrt() * x + beta.sqrt() * eps
        return x_noisy

    def diffuse_all_steps(self, x0):
        '''Simulate and return all forward process steps.'''
        x_noisy = torch.zeros(self.num_steps + 1, *x0.shape)
        x_noisy[0] = x0
        for tidx in range(self.num_steps):
            x_noisy[tidx + 1] = self.diffuse_step(x_noisy[tidx], tidx)
        return x_noisy

    def diffuse(self, x0, tids, return_eps=False):
        '''Simulate different forward process steps.'''
        alpha_bar = self.alphas_bar[tids]
        eps = torch.randn_like(x0)

        missing_shape = [1] * (eps.ndim - alpha_bar.ndim)
        alpha_bar = alpha_bar.view(*alpha_bar.shape, *missing_shape)

        x_noisy = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps

        if return_eps:
            return x_noisy, eps
        else:
            return x_noisy

    def denoise_step(self, x, tids, return_var=False):
        '''Perform single reverse process step.'''
        tids = torch.as_tensor(tids).view(-1, 1) # ensure (batch_size>=1, 1)-shaped tensor
        t = tids + 1 # note that tidx = 0 corresponds to t = 1.0

        eps_pred = self.eps_model(x, t) # predict eps based on noisy x and t

        p = 1 / self.alphas[tids].sqrt()
        q = self.betas[tids] / (1 - self.alphas_bar[tids]).sqrt()

        missing_shape = [1] * (eps_pred.ndim - t.ndim)
        p = p.view(*p.shape, *missing_shape)
        q = q.view(*q.shape, *missing_shape)

        x_denoised_mean = p * (x - q * eps_pred)
        x_denoised_var = self.betas_tilde[tids]
        # x_denoised_var = self.betas[tids]

        if return_var:
            return x_denoised_mean, x_denoised_var
        else:
            return x_denoised_mean

    @torch.no_grad()
    def denoise_all_steps(self, xT):
        '''Perform and return all reverse process steps.'''
        x_denoised = torch.zeros(self.num_steps + 1, *(xT.shape))

        x_denoised[0] = xT
        for idx, tidx in enumerate(reversed(range(self.num_steps))):
            x_denoised_mean, x_denoised_var = self.denoise_step(x_denoised[idx], tidx, return_var=True)

            x_denoised[idx + 1] = x_denoised_mean
            if tidx > 0:
                eps = torch.randn_like(x_denoised[idx + 1])
                x_denoised[idx + 1] += x_denoised_var.sqrt() * eps

        return x_denoised

    @torch.no_grad()
    def generate(self, sample_shape, num_samples=1):
        '''Generate random samples through the reverse process.'''
        x_denoised = torch.randn(num_samples, *sample_shape)

        for tidx in reversed(range(self.num_steps)):
            x_denoised_mean, x_denoised_var = self.denoise_step(x_denoised, tidx, return_var=True)

            x_denoised = x_denoised_mean
            if tidx > 0:
                eps = torch.randn_like(x_denoised)
                x_denoised += x_denoised_var.sqrt() * eps

        return x_denoised

    def loss(self, x):
        '''Compute stochastic loss.'''
        ts = torch.randint(0, self.num_steps, size=(x.shape[0], 1)) # draw random time steps

        x_noisy, eps = self.diffuse(x, ts, return_eps=True)
        eps_pred = self.eps_model(x_noisy, ts) # predict eps based on noisy x and t

        loss = self.criterion(eps_pred, eps)
        return loss

    def training_step(self, batch, batch_idx):
        x_batch = batch[0] # get features and discard labels
        loss = self.loss(x_batch)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

