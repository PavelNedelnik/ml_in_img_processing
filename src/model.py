import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, fst_stride: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=fst_stride, padding=1),
            nn.GroupNorm(8, out_channels), # TODO
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class DecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels, 1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), axis=1)

        x = self.conv(x)
        return x
    

class OutputBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = 3
        self.out = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.out(x)
    

class nnUNet(nn.Module):
    def __init__(self):
        super(nnUNet, self).__init__()

        # encoder
        self.embed1 = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.embed2 = ConvBlock(8, 16, 2)
        self.embed3 = ConvBlock(16, 32, 2)
        self.embed4 = ConvBlock(32, 64, 2)
        self.embed5 = ConvBlock(64, 128, 2)
        self.embed6 = ConvBlock(128, 128, 2)

        # decoder
        self.decode5 = DecoderStage(128, 128)
        self.decode4 = DecoderStage(128, 64)
        self.out4 = OutputBlock(64)
        self.decode3 = DecoderStage(64, 32)
        self.out3 = OutputBlock(32)
        self.decode2 = DecoderStage(32, 16)
        self.out2 = OutputBlock(16)
        self.decode1 = DecoderStage(16, 8)
        self.out1 = OutputBlock(8)

    def forward(self, x):
        x1 = self.embed1(x)
        x2 = self.embed2(x1)
        x3 = self.embed3(x2)
        x4 = self.embed4(x3)
        x5 = self.embed5(x4)
        x = self.embed6(x5)
        x = self.decode5(x, x5)
        x = self.decode4(x, x4)
        y4 = self.out4(x)
        x = self.decode3(x, x3)
        y3 = self.out3(x)
        x = self.decode2(x, x2)
        y2 = self.out2(x)
        x = self.decode1(x, x1)
        y1 = self.out1(x)
        return y1, y2, y3, y4
