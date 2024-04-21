'''Convolutional layers.'''

import torch.nn as nn

from .embed import LearnableSinusoidalEncoding, ClassEmbedding
from .utils import make_conv, make_activation


class DoubleConv(nn.Module):
    '''Double convolution block.'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 bias=True,
                 norm='batch',
                 activation='leaky_relu'):

        super().__init__()

        self.conv_block1 = make_conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
            norm=norm,
            activation=activation
        )

        self.conv_block2 = make_conv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
            norm=norm,
            activation=activation
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x


class CondDoubleConv(DoubleConv):
    '''Double conv. block with position and class conditioning.'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 bias=True,
                 norm='batch',
                 activation='leaky_relu',
                 embed_dim=None,
                 num_classes=None):

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            norm=norm,
            activation=activation
        )

        # create multi-layer positional embedding
        if embed_dim is not None:
            self.time_embed = LearnableSinusoidalEncoding(
                [embed_dim, out_channels, out_channels], # stack two learnable dense layers after the sinusoidal encoding
                activation=activation
            )
        else:
            self.time_embed = None

        # create lookup table class embedding
        if num_classes is not None:
            self.class_embed = ClassEmbedding(num_classes, out_channels)
        else:
            self.class_embed = None

    def forward(self, x, t=None, cids=None):
        out = self.conv_block1(x) # (b, c, h, w)

        # add positional embedding channelwise after first conv block
        if t is not None and self.time_embed is not None:
            t_emb = self.time_embed(t) # (b, c)
            out = out + t_emb.view(*t_emb.shape, 1, 1) # (b, c, h, w)
        elif t is not None and self.time_embed is None:
            raise TypeError('No temporal embedding')
        elif t is None and self.time_embed is not None:
            raise TypeError('No time passed')

        # add class embedding similarly between the convs
        if cids is not None and self.class_embed is not None:
            c_emb = self.class_embed(cids) # (b, c)
            out = out + c_emb.view(*c_emb.shape, 1, 1) # (b, c, h, w)
        elif cids is not None and self.class_embed is None:
            raise TypeError('No class embedding')
        elif cids is None and self.class_embed is not None:
            raise TypeError('No class label passed')

        out = self.conv_block2(out) # (b, c, h, w)
        return out


class ResidualBlock(nn.Module):
    '''Simple residual block.'''

    def __init__(self,
                 num_channels,
                 kernel_size=3, # the classical resblock has a kernel size of 3
                 bias=True,
                 norm='batch',
                 activation='leaky_relu'):

        super().__init__()

        self.conv_block1 = make_conv(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            bias=bias,
            norm=norm,
            activation=activation
        )

        self.conv_block2 = make_conv(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            bias=bias,
            norm=norm,
            activation=None # remove activation from conv block
        )

        self.activation = make_activation(activation) # create separate activation instead

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = out + x # add input before final activation (additive skip connection)
        out = self.activation(out)
        return out


class CondResidualBlock(ResidualBlock):
    '''Residual block with position and class conditioning.'''

    def __init__(self,
                 num_channels,
                 kernel_size=3, # the classical resblock has a kernel size of 3
                 bias=True,
                 norm='batch',
                 activation='leaky_relu',
                 embed_dim=None,
                 num_classes=None):

        super().__init__(
            num_channels,
            kernel_size=kernel_size,
            bias=bias,
            norm=norm,
            activation=activation
        )

        # create multi-layer positional embedding
        if embed_dim is not None:
            self.time_embed = LearnableSinusoidalEncoding(
                [embed_dim, num_channels, num_channels], # stack two learnable dense layers after the sinusoidal encoding
                activation=activation
            )
        else:
            self.time_embed = None

        # create lookup table class embedding
        if num_classes is not None:
            self.class_embed = ClassEmbedding(num_classes, num_channels)
        else:
            self.class_embed = None

    def forward(self, x, t=None, cids=None):
        out = self.conv_block1(x) # (b, c, h, w)

        # add positional embedding channelwise after first conv block
        if t is not None and self.time_embed is not None:
            t_emb = self.time_embed(t) # (b, c)
            out = out + t_emb.view(*t_emb.shape, 1, 1) # (b, c, h, w)
        elif t is not None and self.time_embed is None:
            raise TypeError('No temporal embedding')
        elif t is None and self.time_embed is not None:
            raise TypeError('No time passed')

        # add class embedding similarly between the convs
        if cids is not None and self.class_embed is not None:
            c_emb = self.class_embed(cids) # (b, c)
            out = out + c_emb.view(*c_emb.shape, 1, 1) # (b, c, h, w)
        elif cids is not None and self.class_embed is None:
            raise TypeError('No class embedding')
        elif cids is None and self.class_embed is not None:
            raise TypeError('No class label passed')

        out = self.conv_block2(out) # (b, c, h, w)
        out = out + x # add input before final activation (additive skip connection)
        out = self.activation(out) # (b, c, h, w)
        return out

