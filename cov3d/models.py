import torch
from torch import nn
from torch import Tensor


class ResBlock3d(nn.Module):
    """ 
    Based on
        https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448 
        https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """
    def __init__(self, in_channels:int, out_channels:int, downsample:bool):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
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
        in_channels:int = 1,
        downsample:bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        if downsample:
            self.out_channels *= 2

        self.block1 = ResBlock3d(in_channels=in_channels, out_channels=self.out_channels, downsample=downsample)
        self.block2 = ResBlock3d(in_channels=self.out_channels, out_channels=self.out_channels, downsample=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


class ResNet3dBody(nn.Module):
    def __init__(
        self,
        in_channels:int = 1,
        initial_features:int = 64,
    ):
        super().__init__()

        self.initial_features = initial_features
        self.in_channels = in_channels

        current_num_features = initial_features
        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=current_num_features, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(num_features=current_num_features),
            nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = DownBlock( current_num_features, downsample=True )
        self.layer2 = DownBlock( self.layer1.out_channels, downsample=True )
        self.layer3 = DownBlock( self.layer2.out_channels, downsample=True )
        self.layer4 = DownBlock( self.layer3.out_channels, downsample=True )
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
        num_classes:int=1,
        body = None,
        in_channels:int = 1,
        initial_features:int = 64,
    ):
        super().__init__()
        self.body = body if body is not None else ResNet3dBody(in_channels=in_channels, initial_features=initial_features)
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
