'''U-net architecture.'''

from collections.abc import Sequence

import torch
import torch.nn as nn

from ..layers import CondDoubleConv, CondResidualBlock


class UNet(nn.Module):
    '''
    Conditional U-net.

    Summary
    -------
    A conditional U-net variant is implemented in this module.
    It is composed of encoder, bottleneck and decoder parts.
    While encoder and decoder contain multiple blocks of two normal
    conv-layers, only the bottleneck uses standard residual blocks.

    The time-conditioning is here realized by ingesting a time embedding
    into the middle of such a block, after the first convolution.

    A similar mechanism for class-conditioning is implemented.
    It is based on a lookup table embedding.

    '''

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        bottleneck: nn.Module | None = None
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck

    @classmethod
    def from_params(
        cls,
        in_channels: int = 1,
        mid_channels: Sequence[int] = (16, 32, 64),
        kernel_size: int = 3,
        padding: int = 1,
        norm: str | None = 'batch',
        activation: str | None = 'leaky_relu',
        num_resblocks: int = 3,
        upsample_mode: str = 'conv_transpose',
        embed_dim: int = 128,
        num_classes: int | None = None
    ):
        '''Create instance from architecture parameters.'''

        encoder = UNetEncoder(
            in_channels=in_channels,
            mid_channels=mid_channels,
            kernel_size=kernel_size,
            padding=padding,
            pooling=2,
            norm=norm,
            activation=activation,
            embed_dim=embed_dim,
            num_classes=num_classes
        )

        decoder = UNetDecoder(
            mid_channels=mid_channels[::-1],
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            scaling=2,
            norm=norm,
            activation=activation,
            upsample_mode=upsample_mode,
            embed_dim=embed_dim,
            num_classes=num_classes
        )

        bottleneck = UNetBottleneck(
            num_resblocks=num_resblocks,
            num_channels=mid_channels[-1],
            kernel_size=3,  # fix the kernel size to 3, which is the classical value
            norm=norm,
            activation=activation,
            embed_dim=embed_dim,
            num_classes=num_classes
        )

        return cls(encoder, decoder, bottleneck)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cids: torch.Tensor | None = None
    ) -> torch.Tensor:
        x_list = self.encoder(x, t, cids=cids)

        if self.bottleneck is not None:
            x_list[-1] = self.bottleneck(x_list[-1], t, cids=cids)

        y = self.decoder(x_list, t, cids=cids)
        return y


class UNetEncoder(nn.Module):
    '''Conditional U-net encoder.'''

    def __init__(
        self,
        in_channels: int,
        mid_channels: Sequence[int],
        kernel_size: int = 3,
        padding: int = 1,
        pooling: int = 2,
        norm: str | None = 'batch',
        activation: str | None = 'leaky_relu',
        embed_dim: int | None = None,
        num_classes: int | None = None
    ):
        super().__init__()

        self.first_conv = CondDoubleConv(
            in_channels=in_channels,
            out_channels=mid_channels[0],
            kernel_size=kernel_size,
            padding=padding,
            norm=norm,
            activation=activation,
            embed_dim=embed_dim,
            num_classes=num_classes
        )

        down_list = []
        conv_list = []
        for ch1, ch2 in zip(mid_channels[:-1], mid_channels[1:]):

            down = nn.MaxPool2d(kernel_size=pooling)

            conv = CondDoubleConv(
                in_channels=ch1,
                out_channels=ch2,
                kernel_size=kernel_size,
                padding=padding,
                norm=norm,
                activation=activation,
                embed_dim=embed_dim,
                num_classes=num_classes
            )

            down_list.append(down)
            conv_list.append(conv)

        self.down = nn.ModuleList(down_list)
        self.conv = nn.ModuleList(conv_list)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cids: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        x_list = []

        x = self.first_conv(x, t, cids=cids)
        x_list.append(x)

        for down, conv in zip(self.down, self.conv):
            x = down(x)
            x = conv(x, t, cids=cids)
            x_list.append(x)

        return x_list


class UNetDecoder(nn.Module):
    '''Conditional U-net decoder.'''

    def __init__(
        self,
        mid_channels: Sequence[int],
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        scaling: int = 2,
        norm: str | None = 'batch',
        activation: str | None = 'leaky_relu',
        upsample_mode: str = 'conv_transpose',
        embed_dim: int | None = None,
        num_classes: int | None = None
    ):
        super().__init__()

        up_list = []
        conv_list = []
        for ch1, ch2 in zip(mid_channels[:-1], mid_channels[1:]):

            # bilinear upsampling
            if upsample_mode == 'bilinear':
                up = nn.Upsample(
                    scale_factor=scaling,
                    mode='bilinear',
                    align_corners=True
                )

            # bilinear upsampling followed by a convolution
            elif upsample_mode == 'bilinear_conv':
                up = nn.Sequential(
                    nn.Upsample(
                        scale_factor=scaling,
                        mode='bilinear',
                        align_corners=True
                    ),
                    nn.Conv2d(
                        in_channels=ch1,
                        out_channels=ch2,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding
                    )
                )

            # transposed convolution
            elif upsample_mode == 'conv_transpose':
                up = nn.ConvTranspose2d(
                    in_channels=ch1,
                    out_channels=ch2,
                    kernel_size=scaling,
                    stride=scaling,
                    padding=0
                )

            else:
                raise ValueError('Unknown upsample mode: {}'.format(upsample_mode))

            conv = CondDoubleConv(
                in_channels=2*ch2,  # reserve channels for concatenation skip connection
                out_channels=ch2,
                kernel_size=kernel_size,
                padding=padding,
                norm=norm,
                activation=activation,
                embed_dim=embed_dim,
                num_classes=num_classes
            )

            up_list.append(up)
            conv_list.append(conv)

        self.up = nn.ModuleList(up_list)
        self.conv = nn.ModuleList(conv_list)

        self.last_conv = nn.Conv2d(
            in_channels=mid_channels[-1],
            out_channels=out_channels,
            kernel_size=1,  # use 1x1 convolution in the final layer
            stride=1,
            padding=0
        )

    def forward(
        self,
        x_list: list[torch.Tensor],
        t: torch.Tensor,
        cids: torch.Tensor | None = None
    ) -> torch.Tensor:
        y = x_list[-1]

        for idx, (up, conv) in enumerate(zip(self.up, self.conv)):
            y = up(y)
            y = torch.cat((x_list[-2-idx], y), dim=1)  # concatenate along channel axis
            y = conv(y, t, cids=cids)

        y = self.last_conv(y)
        return y


class UNetBottleneck(nn.Module):
    '''Conditional U-net bottleneck.'''

    def __init__(
        self,
        num_resblocks: int,
        num_channels: int,
        kernel_size: int = 3,  # the classical resblock has a kernel size of 3
        norm: str | None = 'batch',
        activation: str | None = 'leaky_relu',
        embed_dim: int | None = None,
        num_classes: int | None = None
    ):
        super().__init__()

        resblocks_list = []
        for _ in range(num_resblocks):
            resblock = CondResidualBlock(
                num_channels,
                kernel_size=kernel_size,
                norm=norm,
                activation=activation,
                embed_dim=embed_dim,
                num_classes=num_classes
            )

            resblocks_list.append(resblock)

        self.resblocks = nn.ModuleList(resblocks_list)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cids: torch.Tensor | None = None
    ) -> torch.Tensor:
        for resblock in self.resblocks:
            x = resblock(x, t, cids=cids)
        return x
