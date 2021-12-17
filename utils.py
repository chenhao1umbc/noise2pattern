#@title Parts of the U-Net model
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Up1(nn.Module):
    """Special version of Up class with only 1 input
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x1 = self.up(x)
        x2 = self.conv(x1)
        return x2


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64 * factor, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Up_(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
            self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels//2, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class UNetHalf(nn.Module):   # 1 channel, 16*16 to 150*150
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet

        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 512)
        self.up1 = Up_(512, 256, bilinear)
        self.up2 = Up_(256, 128, bilinear)
        self.up3 = Up_(128, 64, bilinear)
        self.up4 = Up_(64, 32, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.ConvTranspose2d(32, 16, kernel_size=5, dilation=3, output_padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, dilation=3, output_padding=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)  # output has W=256, H=256, for gamma = 16
        x = self.reshape(x)
        out = self.outc(x)
        return out


class UNetHalf2(nn.Module):  # 2*2 to 150*150
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet

        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_ch = 512

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(n_channels, self.n_ch//2, bilinear=True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.up5 = Up_(self.n_ch//16, self.n_ch//32, bilinear)
        self.up6 = Up_(self.n_ch//32, self.n_ch//64, bilinear)
        self.up7 = Up_(self.n_ch//64, self.n_ch//128, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//128, self.n_ch//16, kernel_size=3, padding=1, stride=2),
            nn.ConvTranspose2d(self.n_ch//16, 16, kernel_size=5, dilation=3, output_padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, dilation=3, output_padding=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        # x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)  # output has W=256, H=256, for gamma = 16
        x = self.up5(x) # input has W=32, H=32, for gamma = 2
        x = self.up6(x)
        x = self.up7(x)
        x = self.reshape(x) # input 256 output 150
        out = self.outc(x)
        return out


class UNetHalf64_4(nn.Module): # 4*4, 150*150
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet

        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf64_4, self).__init__()
        self.n_ch = 512
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, self.n_ch)
        self.up1 = Up_(self.n_ch, self.n_ch//2, bilinear=True)
        self.up2 = Up_(self.n_ch//2, self.n_ch//4, bilinear)
        self.up3 = Up_(self.n_ch//4, self.n_ch//8, bilinear)
        self.up4 = Up_(self.n_ch//8, self.n_ch//16, bilinear)
        self.up5 = Up_(self.n_ch//16, self.n_ch//32, bilinear)
        self.up6 = Up_(self.n_ch//32, self.n_ch//64, bilinear)
        self.reshape = nn.Sequential(
            nn.Conv2d(self.n_ch//64, self.n_ch//16, kernel_size=3, padding=1, stride=2),
            nn.ConvTranspose2d(self.n_ch//16, 16, kernel_size=5, dilation=3, output_padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, dilation=3, output_padding=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)  # output has W=256, H=256, for gamma = 16
        x = self.up5(x) # input has W=32, H=32, for gamma = 2
        x = self.up6(x)
        x = self.reshape(x) # input 256 output 150
        out = self.outc(x)
        return out

