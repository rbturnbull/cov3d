import numpy as np
import torch
from torch import nn
from torch import Tensor

from fastai.vision.learner import _load_pretrained_weights, _get_first_layer


class ResBlock3d(nn.Module):
    """
    Based on
        https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448
        https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: bool):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.conv1 = nn.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + shortcut
        return self.relu(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        downsample: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        if downsample:
            self.out_channels *= 2

        self.block1 = ResBlock3d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            downsample=downsample,
        )
        self.block2 = ResBlock3d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            downsample=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


class ResNet3dBody(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        initial_features: int = 64,
    ):
        super().__init__()

        self.initial_features = initial_features
        self.in_channels = in_channels

        current_num_features = initial_features
        self.layer0 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=current_num_features,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm3d(num_features=current_num_features),
            nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = DownBlock(current_num_features, downsample=True)
        self.layer2 = DownBlock(self.layer1.out_channels, downsample=True)
        self.layer3 = DownBlock(self.layer2.out_channels, downsample=True)
        self.layer4 = DownBlock(self.layer3.out_channels, downsample=True)
        self.output_features = self.layer4.out_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNet3d(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        body=None,
        in_channels: int = 1,
        initial_features: int = 64,
    ):
        super().__init__()
        self.body = (
            body
            if body is not None
            else ResNet3dBody(
                in_channels=in_channels, initial_features=initial_features
            )
        )
        assert in_channels == self.body.in_channels
        assert initial_features == self.body.initial_features
        self.global_average_pool = torch.nn.AdaptiveAvgPool3d(1)
        self.final_layer = torch.nn.Linear(self.body.output_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.body(x)
        # Final layer
        x = self.global_average_pool(x)
        x = torch.flatten(x, 1)
        output = self.final_layer(x)
        return output


def update_first_layer(model, n_in=1, pretrained=True):
    """Based on fastai function"""
    first_layer, parent, name = _get_first_layer(model)
    assert isinstance(
        first_layer, (nn.Conv2d, nn.Conv3d)
    ), f"Change of input channels only supported with Conv2d or Conv3d, found {first_layer.__class__.__name__}"
    assert (
        getattr(first_layer, "in_channels") == 3
    ), f'Unexpected number of input channels, found {getattr(first_layer, "in_channels")} while expecting 3'
    params = {
        attr: getattr(first_layer, attr)
        for attr in "out_channels kernel_size stride padding dilation groups padding_mode".split()
    }
    params["bias"] = getattr(first_layer, "bias") is not None
    params["in_channels"] = n_in
    new_layer = type(first_layer)(**params)
    if pretrained:
        new_layer.weight.data[:, :1] = first_layer.weight.data.sum(dim=1, keepdim=True)

    setattr(parent, name, new_layer)


class PositionalEncoding3D(nn.Module):
    """
    Adds 3 channels which encode the position of the voxel.

    Requires all tensors to be of the same shape except for the batch dimension
    """

    def __init__(self):
        super().__init__()
        self.start = -2.0
        self.end = 2.0
        self.position_tensor = None

    def forward(self, x):
        shape = x.shape

        # If we haven't created the position tensor yet, then build it here.
        # NB. Requires all tensors to be of the same shape except for the batch dimension
        if self.position_tensor is None:
            mesh = np.mgrid[
                self.start : self.end : shape[2] * 1j,
                self.start : self.end : shape[3] * 1j,
                self.start : self.end : shape[4] * 1j,
            ]
            self.position_tensor = torch.unsqueeze(
                torch.as_tensor(mesh), 0
            ).half()  # hack
            self.position_tensor = self.position_tensor.to(x.device)

        positions = self.position_tensor.repeat(shape[0], 1, 1, 1, 1)

        return torch.cat((x, positions), dim=1)


def adapt_stoic_last_linear_layer(old_linear):
    "Adapts a model built for the stoic dataset to one for the competition dataset"
    if old_linear.out_features == 5:
        return old_linear
        
    new_linear = nn.Linear(in_features=old_linear.in_features, out_features=5, bias=True)    
    new_linear.weight.data[0,:] = old_linear.weight.data[0,:]
    new_linear.weight.data[1,:] = new_linear.weight.data[2,:] = old_linear.weight.data[1,:] * 0.5
    new_linear.weight.data[3,:] = new_linear.weight.data[4,:] = old_linear.weight.data[2,:] * 0.5
    new_linear.bias.data[0] = old_linear.bias.data[0]
    new_linear.bias.data[1] = new_linear.bias.data[2] = old_linear.bias.data[1] * 0.5
    new_linear.bias.data[3] = new_linear.bias.data[4] = old_linear.bias.data[2] * 0.5
    
    return new_linear

def adapt_stoic_model(model):
    "Adapts a model built for the stoic dataset to one for the competition dataset"
    if hasattr(model, "fc"):
        if isinstance(model.fc, nn.Linear):
            model.fc = adapt_stoic_last_linear_layer(model.fc)
        elif isinstance(model.fc, nn.Sequential):
            children = list(model.fc.children())
            children = children[:-1] + [adapt_stoic_last_linear_layer(children[-1])]
            model.fc = nn.Sequential( *children )
    elif hasattr(model, "head"):
        model.head = adapt_stoic_last_linear_layer(model.head)
    