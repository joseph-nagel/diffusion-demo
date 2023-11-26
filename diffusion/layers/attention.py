'''Attention.'''

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    '''
    Self-attention module.

    Summary
    -------
    The self-attention from https://arxiv.org/abs/1805.08318 is implemented.
    It is variant of the classical (scaled) dot product attention,
    which is here applied within a residual skip connection.

    The self-attention mechanism relates different items from a sequence.
    In comparison to conv-layers with limited-size local receptive fields,
    it allows for better capturing global or long-range dependencies.

    '''

    def __init__(self, in_channels, out_channels=None, scaling=False):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels // 8 # set to default value in the paper

        self.f = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) # query
        self.g = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) # key
        self.h = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False) # value

        self.gamma = nn.Parameter(torch.tensor(0.0))

        if scaling:
            scale = torch.tensor(out_channels).sqrt()
            self.register_buffer('scale', scale)
        else:
            self.scale = None

    def forward(self, x):
        b, c, h, w = x.shape

        # flatten tensor (last axis contains the sequence)
        x_flattened = x.view(b, c, h*w) # (b, c, h*w)

        # compute query, key and value
        q = self.f(x_flattened) # (b, c', h*w)
        k = self.g(x_flattened) # (b, c', h*w)
        v = self.h(x_flattened) # (b, c, h*w)

        # compute attention
        attention = torch.bmm(q.transpose(1, 2), k) # (b, h*w, h*w)

        if self.scale is not None:
            attention = attention / self.scale

        attention = torch.softmax(attention, dim=1) # (b, h*w, h*w)
        attention = torch.bmm(v, attention) # (b, c, h*w)

        # add skip connection
        out = self.gamma * attention + x_flattened # (b, c, h*w)

        # reshape
        out = out.view(b, c, h, w) # (b, c, h, w)
        return out

