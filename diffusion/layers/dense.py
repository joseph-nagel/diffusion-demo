'''Fully connected layers.'''

import torch.nn as nn

from .embed import LearnableSinusoidalEncoding
from .utils import make_activation


class ConditionalDense(nn.Module):
    '''Conditional fully connected layer.'''

    def __init__(self,
                 in_features,
                 out_features,
                 activation='relu',
                 embed_dim=None):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

        self.activation = make_activation(activation)

        # create multi-layer positional embedding
        if embed_dim is not None:
            self.emb = LearnableSinusoidalEncoding(
                [embed_dim, out_features, out_features], # stack two learnable dense layers after the sinusoidal encoding
                activation=activation
            )
        else:
            self.emb = None

    def forward(self, x, t):
        out = self.linear(x)

        # add positional embedding (conditioning)
        if self.emb is not None:
            emb = self.emb(t)
            out = out + emb

        if self.activation is not None:
            out = self.activation(out)

        return out


class ConditionalDenseModel(nn.Module):
    '''Conditional fully connected model.'''

    def __init__(self,
                 num_features,
                 activation='relu',
                 embed_dim=None):
        super().__init__()

        if len(num_features) < 2:
            raise ValueError('Number of features needs at least two entries')

        num_layers = len(num_features) - 1

        dense_list = []
        for idx, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            is_not_last = (idx < num_layers - 1)

            dense = ConditionalDense(
                in_features,
                out_features,
                activation=activation if is_not_last else None, # set activation for all layers except the last
                embed_dim=embed_dim # set time embedding for all layers
            )

            dense_list.append(dense)

        self.dense_layers = nn.ModuleList(dense_list)

    def forward(self, x, t):
        for dense in self.dense_layers:
            x = dense(x, t)
        return x

