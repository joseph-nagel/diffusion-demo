'''Learning rate scheduling.'''

import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR


def make_lr_schedule(
    optimizer: Optimizer,
    mode: str | None = 'constant',
    num_total: int | None = None,
    num_warmup: int | None = None,
    last_epoch: int = -1
) -> LRScheduler:
    '''
    Create learning rate scheduler.

    Summary
    -------
    This function creates a learning rate scheduler for an optimizer.
    Constant and cosine schedules with an optional warmup phase are supported.
    The implementation is adapted from `transformers.optimization`.

    Parameters
    ----------
    optimizer : PyTorch optimizer
        Optimizer to apply the learning rate schedule to.
    mode : {'constant', 'cosine'}
        Learning rate schedule type.
    num_total : int
        Total number of steps (for the cosine schedule).
    num_warmup : int
        Number of warmup steps.
    last_epoch : int
        Index of the last epoch (for resuming training).

    '''

    num_warmup = max(0, num_warmup) if num_warmup is not None else 0
    num_total = max(0, num_total) if num_total is not None else 0

    # set constant mode with no warmup
    if mode is None:
        mode = 'constant'

        if num_warmup > 0:
            raise ValueError('Non-zero number of warmup steps')

    # create constant LR scaling function
    if mode == 'constant':
        lr_lambda = partial(_constant_lr_with_warmup, num_warmup=num_warmup)

    # create cosine annealing LR scaling function
    elif mode == 'cosine':
        lr_lambda = partial(
            _cosine_lr_with_warmup,
            num_total=num_total,
            num_warmup=num_warmup
        )

    else:
        raise ValueError(f'Unknown LR schedule type: {mode}')

    # create learning rate scheduler
    lr_scheduler = LambdaLR(
        optimizer,
        lr_lambda,
        last_epoch=last_epoch
    )

    return lr_scheduler


def _constant_lr_with_warmup(current: int, num_warmup: int = 0) -> float:
    '''Return a constant learning rate with a warmup phase.'''

    current = max(0, current)
    num_warmup = max(0, num_warmup)

    if current < num_warmup:
        return current / max(1, num_warmup)
    else:
        return 1.0


def _cosine_lr_with_warmup(
    current: int,
    num_total: int,
    num_warmup: int = 0
) -> float:
    '''Return a cosine annealing learning rate with a warmup phase.'''

    current = max(0, current)
    num_warmup = max(0, num_warmup)
    num_total = max(0, num_total)

    if current < num_warmup:
        return current / max(1, num_warmup)
    elif (current >= num_warmup) and (current <= num_total):
        progress = (current - num_warmup) / (num_total - num_warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError('Current step exceeds total number')
