'''Convolutional layers.'''

import torch.nn as nn
from .embed import SinusoidalEncoding
from .utils import make_conv, make_activation

class DoubleConv(nn.Module):
    '''Double convolution block.'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 norm='batch',
                 activation='relu'):
        super().__init__()

        self.conv_block1 = make_conv(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     norm=norm,
                                     activation=activation)

        self.conv_block2 = make_conv(out_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     norm=norm,
                                     activation=activation)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x


class ConditionalDoubleConv(DoubleConv):
    '''Double convolution block with position conditioning.'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 norm='batch',
                 activation='relu',
                 embed_dim=None):

        super().__init__(in_channels,
                         out_channels,
                         kernel_size=kernel_size,
                         padding=padding,
                         norm=norm,
                         activation=activation)

        if embed_dim is not None:
            self.emb = nn.Sequential(
                SinusoidalEncoding(embed_dim=embed_dim),
                nn.Linear(embed_dim, out_channels)
            )
        else:
            self.emb = None

    def forward(self, x, t):
        out = self.conv_block1(x)

        # add positional embedding channelwise after first conv block (conditioning)
        if self.emb is not None:
            emb = self.emb(t)
            out = out + emb.view(*emb.shape, 1, 1)

        out = self.conv_block2(out)
        return out


class ResidualBlock(nn.Module):
    '''Simple residual block.'''

    def __init__(self,
                 num_channels,
                 kernel_size=3, # the classical resblock has a kernel size of 3
                 norm='batch',
                 activation='relu'):
        super().__init__()

        self.conv_block1 = make_conv(num_channels,
                                     num_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding='same',
                                     norm=norm,
                                     activation=activation)

        self.conv_block2 = make_conv(num_channels,
                                     num_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding='same',
                                     norm=norm,
                                     activation=None) # remove activation from conv block
        self.activation = make_activation(activation) # create separate activation instead

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = out + x # add input before final activation (additive skip connection)
        out = self.activation(out)
        return out


class ConditionalResidualBlock(ResidualBlock):
    '''Residual block with position conditioning.'''

    def __init__(self,
                 num_channels,
                 kernel_size=3, # the classical resblock has a kernel size of 3
                 norm='batch',
                 activation='relu',
                 embed_dim=None):

        super().__init__(num_channels,
                         kernel_size=kernel_size,
                         norm=norm,
                         activation=activation)

        if embed_dim is not None:
            self.emb = nn.Sequential(
                SinusoidalEncoding(embed_dim=embed_dim),
                nn.Linear(embed_dim, num_channels)
            )
        else:
            self.emb = None

    def forward(self, x, t):
        out = self.conv_block1(x)

        # add positional embedding channelwise after first conv block (conditioning)
        if self.emb is not None:
            emb = self.emb(t)
            out = out + emb.view(*emb.shape, 1, 1)

        out = self.conv_block2(out)
        out = out + x # add input before final activation (additive skip connection)
        out = self.activation(out)
        return out
