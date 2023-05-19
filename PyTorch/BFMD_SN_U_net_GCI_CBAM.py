import torch
import torch.nn as nn
import torch.nn.functional as F

from block_feature_map_distortion import *
from switchable_norm_2d import *
from bfmd_sn_u_net_gci_cbam_blocks import *
from GCI_CBAM import *


class BFMD_SN_UNet_with_GCI_CBAM(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(BFMD_SN_UNet_with_GCI_CBAM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = (DoubleConv_BFMD_SN_Unet(n_channels, 16,SN=False,Distortion=False))
        self.down1 = (Down(16, 32,SN=False,Distortion=False))
        self.down2 = (Down(32, 64,SN=False,Distortion=False))
        self.down3 = (Down(64, 128,SN=False,Distortion=False))

        self.up1 = (Up(128, 64,SN=False,Distortion=False)) # 320
        self.up2 = (Up(64, 32,SN=False,Distortion=False)) # 160
        self.up3 = (Up(32, 16,SN=False,Distortion=False)) # 80
        self.outc = (OutConv(16, n_classes))
        self.channel = GCI_ChannelAttention(1)
        self.spatial = GCI_SpatialAttention()

    def forward(self, x):
        x1 = self.inc(x)
        x1c,x2c,x3c,x4c = self.channel(x1)
        x1sc,x2sc,x3sc,x4sc = self.spatial(x1c,x2c,x3c,x4c)
        x1_gci = torch.cat([x1sc, x2sc, x3sc, x4sc], dim=1)
        
        x2 = self.down1(x1)
        x1c,x2c,x3c,x4c = self.channel(x2)
        x1sc,x2sc,x3sc,x4sc = self.spatial(x1c,x2c,x3c,x4c)
        x2_gci = torch.cat([x1sc, x2sc, x3sc, x4sc], dim=1)
        
        x3 = self.down2(x2)
        x1c,x2c,x3c,x4c = self.channel(x3)
        x1sc,x2sc,x3sc,x4sc = self.spatial(x1c,x2c,x3c,x4c)
        x3_gci = torch.cat([x1sc, x2sc, x3sc, x4sc], dim=1)

        x4 = self.down3(x3)

        x = self.up1(x4, x3_gci)
        x = self.up2(x, x2_gci)
        x = self.up3(x, x1_gci)
        pred = self.outc(x)
        return pred

model = BFMD_SN_UNet_with_GCI_CBAM(n_channels=3, n_classes=1)
