'''Modeling utilities.'''

import torch.nn as nn


def make_dense(in_features,
               out_features,
               bias=True,
               activation=None):
    '''
    Create fully connected layer.

    Parameters
    ----------
    in_features : int
        Number of inputs.
    out_features : int
        Number of outputs.
    bias : bool
        Determines whether a bias is used.
    activation : None or str
        Determines the nonlinearity.

    '''

    linear = nn.Linear(in_features, out_features, bias=bias)
    activation = make_activation(activation)

    layers = [linear, activation]
    dense_block = nn.Sequential(*layers)

    return dense_block


def make_conv(in_channels,
              out_channels,
              kernel_size=3,
              stride=1,
              padding=1,
              bias=True,
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
    bias : bool
        Determines whether a bias is used.
    norm : None or str
        Determines the normalization.
    activation : None or str
        Determines the nonlinearity.

    '''

    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias # the bias should be disabled if a batchnorm directly follows after the convolution
    )

    activation = make_activation(activation)
    norm = make_norm(norm, num_features=out_channels)

    layers = [conv, activation, norm] # note that the normalization follows the activation (which could be reversed of course)
    conv_block = nn.Sequential(*layers)

    return conv_block


def make_activation(mode):
    '''Create activation.'''
    if mode is None or mode == 'none':
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


def make_norm(mode, num_features):
    '''Create normalization.'''
    if mode is None or mode == 'none':
        norm = nn.Identity()
    elif mode == 'batch':
        norm = nn.BatchNorm2d(num_features)
    elif mode == 'instance':
        norm = nn.InstanceNorm2d(num_features)
    else:
        raise ValueError('Unknown normalization type: {}'.format(mode))
    return norm

