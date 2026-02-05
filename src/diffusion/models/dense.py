'''Fully connected model.'''

from collections.abc import Sequence

import torch
import torch.nn as nn

from ..layers import CondDense


class CondDenseModel(nn.Module):
    '''Conditional fully connected model.'''

    def __init__(
        self,
        num_features: Sequence[int],
        activation: str | None = 'leaky_relu',
        embed_dim: int | None = None
    ):
        super().__init__()

        if len(num_features) < 2:
            raise ValueError('Number of features needs at least two entries')

        num_layers = len(num_features) - 1

        dense_list = []

        for idx, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            is_not_last = (idx < num_layers - 1)

            dense = CondDense(
                in_features,
                out_features,
                activation=activation if is_not_last else None,  # set activation for all layers except the last
                embed_dim=embed_dim  # set time embedding for all layers
            )

            dense_list.append(dense)

        self.dense_layers = nn.ModuleList(dense_list)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cids: torch.Tensor | None = None
    ) -> torch.Tensor:

        if cids is not None:
            raise NotImplementedError('Class conditioning is not implemented')

        for dense in self.dense_layers:
            x = dense(x, t)

        return x
