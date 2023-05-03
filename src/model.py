import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
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
    

class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        skip = self.conv(x)
        return skip, self.down(skip)


class DecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(2 * in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x) # TODO
        diffZ = skip.shape[2] - x.shape[2]
        diffY = skip.shape[3] - x.shape[3]
        diffX = skip.shape[4] - x.shape[4]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat((x, skip), axis=1)
        return self.conv(x)
    

class nnUNet(nn.Module):
    def __init__(self):
        super(nnUNet, self).__init__()
        # encoder
        self.embed1 = EncoderStage(4, 16)
        self.embed2 = EncoderStage(16, 32)
        self.embed3 = EncoderStage(32, 64)
        self.embed4 = EncoderStage(64, 128)
        self.embed5 = EncoderStage(128, 128)
        self.bottleneck = ConvBlock(128, 128)

        # decoder
        self.decode5 = DecoderStage(128, 64)
        self.decode4 = DecoderStage(64, 64)
        self.out4 = nn.Conv3d(64, 3, kernel_size=1)
        self.decode3 = DecoderStage(64, 32)
        self.out3 = nn.Conv3d(32, 3, kernel_size=1)
        self.decode3 = DecoderStage(32, 16)
        self.out2 = nn.Conv3d(16, 3, kernel_size=1)
        self.decode1 = DecoderStage(16, 8)
        self.out1 = nn.Conv3d(8, 3, kernel_size=1)

    def forward(self, x):
        x1, x = self.embed1(x)
        x2, x = self.embed2(x)
        x3, x = self.embed3(x)
        x4, x = self.embed4(x)
        x5, x = self.embed5(x)
        x = self.bottleneck(x)
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
