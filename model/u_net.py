from torch import nn
from torch.nn import functional as F


def conv_3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, padding=1)


class Conv3BN(nn.Module):
    """A module which applies the following actions:
        - convolution with 3x3 kernel;
        - batch normalization (if enabled);
        - ReLU.

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        batch_normalization: A boolean indicating if Batch Normalization is enabled or not.
    """

    def __init__(self, in_channels: int, out_channels: int, batch_normalization=True):
        super().__init__()
        self.conv = conv_3x3(in_channels, out_channels)
        self.batch_normalization = nn.BatchNorm2d(out_channels) if batch_normalization else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_normalization is not None:
            x = self.batch_normalization(x)
        x = self.activation(x)
        return x


class UNetModule(nn.Module):
    """A module which applies Conv3BN `num_layers` times.
    """

    def __init__(self, in_channels: int, out_channels: int, num_layers: int=2):
        super().__init__()
        self.conv_layers = nn.ModuleList([Conv3BN(in_channels, out_channels)])
        for dummy_index in range(num_layers - 1):
            self.conv_layers.append(Conv3BN(out_channels, out_channels))

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x


class UNet(nn.Module):
    """A UNet module. Applies
        - `depth` times of
            - UNetModule
            - MaxPool2d

        - UNetModule

        - `depth` times of
            - UpsamplingNearest2d
            - UNetModule

        - Conv2d
        - activation (sigmoid)

        The number of output channels of each UNetModule is twice larger/less than the previous
        number of input channels (except for the first UNetModule);

    Arguments:
        num_classes: Number of output channels.
        base_in_channels: Number of input image channels.
        base_out_channels: Number of out channels of the first UNet layer (UNetModule).
        depth: number of UNet layers (UNetModule) on the way down. Same number on the way up.
    """

    def __init__(self,
                 num_classes: int=1,
                 base_in_channels: int=3,
                 base_out_channels: int=16,
                 depth: int=6):
        super().__init__()
        down_out_channels = [base_out_channels * 2**i for i in range(depth)]
        center_out_channels = base_out_channels * 2**depth
        up_out_channels = list(reversed(down_out_channels))

        self.layers = nn.ModuleList()
        print(self.layers)
        in_channels = base_in_channels

        # Going down:
        for out_channels in down_out_channels:
            u_net_module = UNetModule(in_channels, out_channels, num_layers=2)
            self.layers.append(u_net_module)
            self.layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        # Center:
        self.layers.append(UNetModule(in_channels, center_out_channels, num_layers=2))
        in_channels = center_out_channels

        # Going up:
        for out_channels in up_out_channels:
            self.layers.append(nn.UpsamplingNearest2d(scale_factor=2))
            self.layers.append(UNetModule(in_channels, out_channels, num_layers=3))
            in_channels = out_channels

        # Final layer and activation:
        self.layers.append(nn.Conv2d(in_channels, out_channels=num_classes, kernel_size=1))
        self.activation = F.sigmoid

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        x = self.activation(x)
        return x
