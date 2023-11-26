'''Beta schedules.'''

import torch


def make_beta_schedule(num_steps,
                       mode='cosine',
                       beta_range=(1e-04, 0.02),
                       cosine_s=0.008,
                       sigmoid_range=(-5, 5)):
    '''
    Create beta schedule.

    Summary
    -------
    This function allows one to create different beta schedules.
    Simple linear and quadratic schemes are implemented,
    for which one needs to specify the corresponding beta range.

    Alternatively, one may impose a schedule on the alpha_bar parameters.
    In the cosine-based approach from https://arxiv.org/abs/2102.09672,
    the betas are calculated for predefined values of alpha_bar.

    An analogous sigmoid-based approach is also implemented.
    In contrast to the approach in https://arxiv.org/abs/2212.11972,
    the sigmoid-curve is assigned to the square root of alpha_bar.

    Parameters
    ----------
    num_steps : int
        Number of time steps.
    mode : {'linear', 'quadratic', 'cosine', 'sigmoid'}
        Determines the scheduling type.
    beta_range : (float, float)
        Beta range for linear and quadratic schedules.
    cosine_s : float
        Offset parameter for cosine-based alpha_bar.
    sigmoid_range : (float, float)
        Input value range the sigmoid is evaluated for
        in the corresponding sqrt.(alpha_bar) schedule.

    '''

    # linear beta
    if mode == 'linear':
        if len(beta_range) == 2:
            beta_start, beta_end = beta_range
        else:
            raise ValueError('Beta range should have two entries')

        if all([(beta_bound > 0 and beta_bound < 1) for beta_bound in beta_range]):
            betas = torch.linspace(beta_start, beta_end, steps=num_steps)
        else:
            raise ValueError('Invalid beta range encountered')

    # quadratic beta
    elif mode == 'quadratic':
        if len(beta_range) == 2:
            beta_start, beta_end = beta_range
        else:
            raise ValueError('Beta range should have two entries')

        if all([(beta_bound > 0 and beta_bound < 1) for beta_bound in beta_range]):
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, steps=num_steps)**2
        else:
            raise ValueError('Invalid beta range encountered')

    # cosine-based alpha_bar
    elif mode == 'cosine':
        cosine_s = abs(cosine_s)

        ts = torch.arange(num_steps + 1)
        alphas_bar = torch.cos((ts / num_steps + cosine_s) / (1 + cosine_s) * torch.pi / 2)**2
        alphas_bar = alphas_bar / alphas_bar.max()

        betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
        betas = torch.clip(betas, 0.0001, 0.9999)

    # sigmoid-based sqrt.(alpha_bar)
    elif mode == 'sigmoid':
        if len(sigmoid_range) == 2:
            ts = torch.linspace(*sigmoid_range, num_steps + 1)
            alphas_bar = torch.sigmoid(-ts)**2
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        else:
            raise ValueError('Sigmoid range should have two entries')

    else:
        raise ValueError('Unknown schedule type: {}'.format(mode))

    return betas

