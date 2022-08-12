import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

class DownBlock(nn.Module):
    # 2 convolutions and then downsampling (pooling)

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=1)
        if pooling == True:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pooling = x
        if self.pooling:
            x = self.pool(x)

        return before_pooling, x

class UpBlock(nn.Module):
    # upsampling and then 2 convolutions

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = nn.Upsample(mode='bilinear', scale_factor=2),
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True, groups=1)
        self.conv2 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=1)

        def forward(self, x_up, x_down):
            # upsampling unit
            x = self.upsample(x_up)
            x = F.relu(self.conv1(x))
            x = torch.cat((x, x_down), 1)

            # conv unit
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))

            return x

class UNet(nn.Module):
    def __init__(self, num_classes, in_channels, depth, seed_filters):
        super(UNet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.depth = depth
        self.seed_filters = seed_filters

        self.upblocks = []
        self.downblocks = []

        num_filters_in = self.in_channels
        num_filters_out = self.seed_filters
        self.upblocks.append(DownBlock(in_channels=num_filters_in, out_channels=num_filters_out, pooling=True))
        for i in range(1, depth):
            pooling = True
            if i == depth-1:
                pooling=False
            num_filters_in=num_filters_out
            num_filters_out *= 2

            down_block = DownBlock(in_channels=num_filters_in, out_channels=num_filters_out, pooling=pooling)
            self.downblocks.append(down_block)

        for i in range(1, depth):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            up_block = UpBlock(in_channels=num_filters_in, out_channels=num_filters_out)
            self.upblocks.append(up_block)
        
        self.upblocks = nn.ModuleList(self.upblocks)
        self.downblocks = nn.ModuleList(self.downblocks)
        self.conv1 = nn.Conv2d(num_filters_out, self.num_classes, kernel_size=1, stride=1, padding=1, bias=True, groups=1)

    def forward(self, x):
        encoder_output_stagewise = []
        for module in self.downblocks:
            before_pooling, x = module(x)
            encoder_output_stagewise.append(before_pooling)
            
        for i, module in enumerate(self.upblocks):
            x = module(x, encoder_output_stagewise[-(i+2)])
        
        x = self.conv1(x)

        return x