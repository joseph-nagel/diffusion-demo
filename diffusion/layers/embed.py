'''Positional embedding.'''

import torch
import torch.nn as nn

from .utils import make_dense


class SinusoidalEncoding(nn.Module):
    '''
    Sinusoidal position encoding.

    Summary
    -------
    This class implements the embedding from the transformer paper https://arxiv.org/abs/1706.03762.
    It can be used in order to encode spatial positions or times and ingest them in further layers.
    For a (batch_size, 1)-shaped input, the (batch_size, embed_dim)-sized embedding is computed.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the embedding space.

    '''

    def __init__(self, embed_dim):
        super().__init__()

        # set embedding dimension
        embed_dim = abs(embed_dim)

        if embed_dim < 2:
            raise ValueError('At least two embedding dimensions required')
        elif embed_dim % 2 != 0:
            raise ValueError('Dimensionality has to be an even number')

        self.embed_dim = embed_dim

        # create angular frequencies
        omega = self._make_frequencies()
        self.register_buffer('omega', omega)

    def _make_frequencies(self):
        '''Create angular frequencies.'''
        i = torch.arange(self.embed_dim // 2).view(1, -1)
        omega = 1 / (10000 ** (2 * i / self.embed_dim))
        return omega

    def forward(self, t):
        # ensure (batch_size>=1, 1)-shaped tensor
        if t.numel() == 1:
            t = t.view(1, 1)
        elif t.ndim != 2 or t.shape[1] != 1:
            raise ValueError('Invalid shape encountered: {}'.format(t.shape))

        device = t.device
        batch_size = t.shape[0]

        emb = torch.zeros(batch_size, self.embed_dim, device=device)
        emb[:,0::2] = torch.sin(self.omega * t)
        emb[:,1::2] = torch.cos(self.omega * t)
        return emb


class LearnableSinusoidalEncoding(nn.Sequential):
    '''
    Learnable sinusoidal position encoding.

    Summary
    -------
    Multiple FC layers are stacked after a sinusoidal encoding.
    This represents a learnable variant of the position embedding.

    Parameters
    ----------
    num_features : list of ints
        List of layer-specific feature numbers.
    activation : None or str
        Determines the nonlinearity for all layers but the last.

    '''

    def __init__(self, num_features, activation='relu'):

        if len(num_features) < 2:
            raise ValueError('Number of features needs at least two entries')

        num_dense_layers = len(num_features) - 1

        # create sinusoidal encoding
        embed_dim = num_features[0]
        sinusoidal_encoding = SinusoidalEncoding(embed_dim=embed_dim)

        # create FC layers
        dense_list = []
        for idx, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            is_not_last = (idx < num_dense_layers - 1)

            dense = make_dense(
                in_features,
                out_features,
                activation=activation if is_not_last else None # set activation for all layers except the last
            )

            dense_list.append(dense)

        super().__init__(sinusoidal_encoding, *dense_list)

