import torch
import torch.nn as nn

class DoubleConv_BFMD_SN_Unet(nn.Module):

    def __init__(self, in_channels, out_channels,distortion_probabilty,feature_num,SN=False,Distortion=False,mid_channels=None,):
        super().__init__()
        if SN and not Distortion:
          if not mid_channels:
              mid_channels = out_channels
          self.double_conv = nn.Sequential(
              nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
              Switch_able_Normalization(feature_num),
              nn.ReLU(inplace=True),
              nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
              Switch_able_Normalization(feature_num),
              nn.ReLU(inplace=True)
          )
        if Distortion and not SN:
          if not mid_channels:
              mid_channels = out_channels
          self.double_conv = nn.Sequential(
              nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
              BFMD(dist_prob=distortion_probabilty),
              nn.ReLU(inplace=True),
              nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
              BFMD(dist_prob=distortion_probabilty),
              nn.ReLU(inplace=True)
          )
        if Distortion and SN:
          if not mid_channels:
              mid_channels = out_channels
          self.double_conv = nn.Sequential(
              nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
              BFMD(dist_prob=distortion_probabilty),
              Switch_able_Normalization(feature_num),
              nn.ReLU(inplace=True),
              nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
              BFMD(dist_prob=distortion_probabilty),
              Switch_able_Normalization(feature_num),
              nn.ReLU(inplace=True)
          )
        if not Distortion and not SN:
          if not mid_channels:
              mid_channels = out_channels
          self.double_conv = nn.Sequential(
              nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
              nn.ReLU(inplace=True),
              nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
              nn.ReLU(inplace=True)
              )


    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,distortion_probabilty,feature_num,SN,Distortion):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_BFMD_SN_Unet(in_channels, out_channels,distortion_probabilty,feature_num,SN=SN,Distortion=Distortion)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,distortion_probabilty,feature_num,SN,Distortion):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2)
        self.conv = DoubleConv_BFMD_SN_Unet(in_channels, out_channels,distortion_probabilty,feature_num,SN=SN,Distortion=Distortion)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        return self.sigmoid(x)

