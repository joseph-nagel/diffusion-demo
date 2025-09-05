'''Fully connected layers.'''

import torch
import torch.nn as nn

from .embed import LearnableSinusoidalEncoding
from .utils import make_activation


class CondDense(nn.Module):
    '''Conditional fully connected layer.'''

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str | None = 'leaky_relu',
        embed_dim: int | None = None
    ):

        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

        self.activation = make_activation(activation)

        # create multi-layer positional embedding
        if embed_dim is not None:
            self.emb = LearnableSinusoidalEncoding(
                [embed_dim, out_features, out_features],  # stack two learnable dense layers after the sinusoidal encoding
                activation=activation
            )
        else:
            self.emb = None

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        out = self.linear(x)

        # add positional embedding (conditioning)
        if t is not None and self.emb is not None:
            emb = self.emb(t)
            out = out + emb
        elif t is not None and self.emb is None:
            raise TypeError('No temporal embedding')
        elif t is None and self.emb is not None:
            raise TypeError('No time passed')

        if self.activation is not None:
            out = self.activation(out)

        return out
