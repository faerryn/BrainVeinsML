import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=True):
        super(DoubleConv, self).__init__()
        if mid_channels:
            mid_channels = out_channels // 2
        print(f"mid channels {mid_channels}")
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, 1, 1, bias=True),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(),
            nn.Conv3d(mid_channels, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, input):
        return self.conv(input)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, input):
        return self.encoder(input)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 = self.up(x1)
        # print(f"in up conv x1 {x1.shape} x2 {x2.shape}")
        x = torch.cat((self.up(x1), x2), dim=1)
        # print(f"in up2 conv x {x.shape}")
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, 1, 1, bias=True),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(),
            nn.Conv3d(mid_channels, mid_channels, 3, 1, 1, bias=True),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=True):
        super(UNET, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.firstConv = DoubleConv(n_channels, 64)
        self.encoder1 = Down(64, 128)
        self.encoder2 = Down(128, 256)
        self.encoder3 = Down(256, 512)
        # self.encoder4 = Down(512, 1024)

        self.decoder1 = Up(512, 256)
        self.decoder2 = Up(256, 128)
        self.decoder3 = Up(128, 64)
        self.outconv = OutConv(64, 2)

    def forward(self, x):
        torch.cuda.empty_cache()
        # print(f"x {x.shape}")
        x1 = self.firstConv(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        # print(f"x1 {x1.shape} x2 {x2.shape} x3 {x3.shape} x4 {x4.shape}")

        x5 = self.decoder1(x4, x3)
        x6 = self.decoder2(x5, x2)
        # print(f"x5 {x5.shape} x6 {x6.shape}")
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # print("Using {} device".format(device))
        x7 = self.decoder3(x6, x1)
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # print("Using {} device".format(device))
        final = self.outconv(x7)
        # print(f"x5 {x5.shape} x6 {x6.shape} x7 {x7.shape} final {final.shape}")

        return final
