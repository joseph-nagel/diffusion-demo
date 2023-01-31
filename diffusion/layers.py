'''Model layers.'''

import torch
import torch.nn as nn

def make_conv(in_channels,
              out_channels,
              kernel_size=3,
              stride=1,
              padding=1,
              norm=None,
              activation=None):
    '''
    Create convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolutional kernel size.
    stride : int
        Stride parameter.
    padding : int
        Padding parameter.
    norm : None or str
        Determines the normalization.
    activation : None or str
        Determines the nonlinearity.

    '''

    conv = nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     bias=norm != 'batch') # disable bias when using batchnorm

    activation = make_activation(activation)
    norm = make_norm(norm, num_features=out_channels)

    layers = [conv, activation, norm]
    # layers = [l for l in layers if l is not None]
    conv_block = nn.Sequential(*layers)

    return conv_block


def make_norm(mode, num_features):
    '''Create normalization.'''
    if mode is None:
        norm = nn.Identity()
    elif mode == 'batch':
        norm = nn.BatchNorm2d(num_features)
    elif mode == 'instance':
        norm = nn.InstanceNorm2d(num_features)
    else:
        raise ValueError('Unknown normalization type: {}'.format(mode))
    return norm


def make_activation(mode):
    '''Create activation.'''
    if mode is None:
        activation = nn.Identity()
    elif mode == 'sigmoid':
        activation = nn.Sigmoid()
    elif mode == 'tanh':
        activation = nn.Tanh()
    elif mode == 'relu':
        activation = nn.ReLU()
    elif mode == 'leaky_relu':
        activation = nn.LeakyReLU()
    elif mode == 'elu':
        activation = nn.ELU()
    elif mode == 'softplus':
        activation = nn.Softplus()
    elif mode == 'swish':
        activation = nn.SiLU()
    else:
        raise ValueError('Unknown activation function: {}'.format(mode))
    return activation


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
            # print('t:', t.shape)
            # print('out:', out.shape)
            # print('emb:', emb.shape)
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
            # print('t:', t.shape)
            # print('out:', out.shape)
            # print('emb:', emb.shape)
            out = out + emb.view(*emb.shape, 1, 1)

        out = self.conv_block2(out)
        out = out + x # add input before final activation (additive skip connection)
        out = self.activation(out)
        return out


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

