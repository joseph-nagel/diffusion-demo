'''Attention.'''

import torch
import torch.nn as nn


class SelfAttention2D(nn.Module):
    '''
    Self-attention with skip connections for 2D data.

    Summary
    -------
    This module establishes the self-attention from https://arxiv.org/abs/1805.08318.
    It employs a residual skip connection adding the inputs after the attention.
    The input shape for this layer is (batch, channels, height, width).

    Parameters
    ----------
    in_channels : int
        Number of input and output channels.
    out_channels : int or None
        Number of queries and keys.
    scale : bool
        Determines whether scores are scaled.

    '''

    def __init__(
        self,
        in_channels,
        out_channels=None,
        scale=False
    ):

        super().__init__()

        if out_channels is None:
            out_channels = in_channels // 8  # set to the default value in the paper

        self.f = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)  # query
        self.g = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)  # key
        self.h = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)  # value

        self.gamma = nn.Parameter(torch.tensor(0.0))

        if scale:
            d_k_sqrt = torch.tensor(out_channels).sqrt()
            self.register_buffer('scale', d_k_sqrt)
        else:
            self.scale = None

    def forward(self, x):

        b, c, h, w = x.shape

        # flatten tensor (last axis contains the sequence)
        x_flattened = x.view(b, c, h*w)  # (b, c, h*w)

        # compute query, key and value
        q = self.f(x_flattened)  # (b, c', h*w)
        k = self.g(x_flattened)  # (b, c', h*w)
        v = self.h(x_flattened)  # (b, c, h*w)

        # compute attention
        algn_scores = torch.bmm(q.transpose(1, 2), k)  # (b, h*w, h*w)

        if self.scale is not None:
            algn_scores = algn_scores / self.scale

        attn_weights = torch.softmax(algn_scores, dim=1)  # (b, h*w, h*w)

        attention = torch.bmm(v, attn_weights)  # (b, c, h*w)

        # add skip connection
        out = self.gamma * attention + x_flattened  # (b, c, h*w)

        # reshape
        out = out.view(b, c, h, w)  # (b, c, h, w)

        return out

