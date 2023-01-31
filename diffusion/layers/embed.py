'''Positional embedding.'''

import torch
import torch.nn as nn

class SinusoidalEncoding(nn.Module):
    '''
    Sinusoidal position encoding.

    Summary
    -------
    This class implements the embedding from the attention paper https://arxiv.org/abs/1706.03762.
    It can be used in order to encode spatial positions or times and ingest them in further layers.
    For a (batch_size, 1)-shaped input, the (batch_size, embed_dim)-sized embedding is computed.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the embedding space.

    '''

    def __init__(self, embed_dim):
        super().__init__()

        if embed_dim % 2 != 0:
            raise ValueError('Uneven dimensionality requested: {}'.format(embed_dim))

        self.embed_dim = embed_dim

    def forward(self, t):
        # ensure (batch_size>=1, 1)-shaped tensor
        if t.size == 1:
            t = t.view(1, 1)
        elif t.ndim != 2 or t.shape[1] != 1:
            raise ValueError('Invalid shape encountered: {}'.format(t.shape))

        device = t.device
        batch_size = t.shape[0]

        i = torch.arange(self.embed_dim // 2, device=device).view(1, -1)
        aux = t / (10000 ** (2*i / self.embed_dim))

        emb = torch.zeros(batch_size, self.embed_dim, device=device)
        emb[:,0::2] = torch.sin(aux)
        emb[:,1::2] = torch.cos(aux)
        return emb

