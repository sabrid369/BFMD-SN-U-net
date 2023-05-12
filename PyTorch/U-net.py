from block_feature_map_distortion import *
from switchable_norm_2d import *
from bfmd_sn_unet_blocks import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = (DoubleConv_BFMD_SN_Unet(n_channels, 16,distortion_probabilty=0.1,feature_num=16,SN=False,Distortion=False))
        self.down1 = (Down(16, 32,distortion_probabilty= 0.1,feature_num=32,SN=False,Distortion=False))
        self.down2 = (Down(32, 64,distortion_probabilty=0.1,feature_num=64,SN=False,Distortion=False))
        self.down3 = (Down(64, 128,distortion_probabilty=0.1,feature_num=128,SN=False,Distortion=False))

        self.up1 = (Up(128, 64 ,distortion_probabilty=0.1,feature_num=64,SN=False,Distortion=False))
        self.up2 = (Up(64, 32,distortion_probabilty=0.1,feature_num=32,SN=False,Distortion=False))
        self.up3 = (Up(32, 16,distortion_probabilty=0.1,feature_num=16,SN=False,Distortion=False))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        pred = self.outc(x)
        return pred

model = UNet(n_channels=3, n_classes=1)
model = model.to(memory_format=torch.channels_last)
model
