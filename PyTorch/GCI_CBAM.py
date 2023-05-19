import torch
import torch.nn as nn
import torch.nn.functional as F

class GCI_ChannelAttention(nn.Module):
    def __init__(self, in_feature):
        super(GCI_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        ks=3
        self.fc1 = nn.Conv1d(in_feature,1, 3,dilation=1,padding=(ks - 1) // 2,bias=False)
        self.fc2 = nn.Conv1d(in_feature, 1, 3,dilation=2,padding=(ks+2 - 1) // 2,bias=False)
        self.fc3 = nn.Conv1d(in_feature, 1, 3,dilation=3,padding=(ks+2+2 - 1) // 2,bias=False)
        self.fc4 = nn.Conv1d(in_feature, 1, 3,dilation=4,padding=(ks+2+2+2- 1) // 2,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        
        avg_out1 = (self.fc1(avg_out.squeeze(-1).permute(0,2,1))).permute(0,2,1).unsqueeze(-1)
        max_out1 = (self.fc1(max_out.squeeze(-1).permute(0,2,1))).permute(0,2,1).unsqueeze(-1)
        out1 = avg_out1 + max_out1

        avg_out2 = (self.fc2(avg_out.squeeze(-1).permute(0,2,1))).permute(0,2,1).unsqueeze(-1)
        max_out2 = (self.fc2(max_out.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1))
        out2 = avg_out2 + max_out2

        avg_out3 = (self.fc3(avg_out.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1))
        max_out3 = (self.fc3(max_out.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1))
        out3 = avg_out3 + max_out3

        avg_out4 = (self.fc4(avg_out.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1))
        max_out4 = (self.fc4(max_out.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1))

        out4 = avg_out4 + max_out4

        out1 = self.sigmoid(out1)
        out2 = self.sigmoid(out2)
        out3 = self.sigmoid(out3)
        out4 = self.sigmoid(out4)
        return out1.expand_as(x)*x,out2.expand_as(x)*x,out3.expand_as(x)*x,out4.expand_as(x)*x

class GCI_SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(GCI_SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2,dilation=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size+5)//2,dilation=2)
        self.conv3 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size+5+7)//2,dilation=3)
        self.conv4 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size+5+7+5)//2,dilation=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1,x2,x3,x4):
        avg_out1 = torch.mean(x1, dim=1, keepdim=True)
        max_out1, _ = torch.max(x1, dim=1, keepdim=True)

        avg_out2 = torch.mean(x2, dim=1, keepdim=True)
        max_out2, _ = torch.max(x2, dim=1, keepdim=True)

        avg_out3 = torch.mean(x3, dim=1, keepdim=True)
        max_out3, _ = torch.max(x3, dim=1, keepdim=True)

        avg_out4 = torch.mean(x4, dim=1, keepdim=True)
        max_out4, _ = torch.max(x4, dim=1, keepdim=True)

        x1_ = torch.cat([avg_out1, max_out1], dim=1)
        x2_ = torch.cat([avg_out2, max_out2], dim=1)
        x3_ = torch.cat([avg_out3, max_out3], dim=1)
        x4_ = torch.cat([avg_out4, max_out4], dim=1)

        x1_ = self.conv1(x1_)
        x2_ = self.conv2(x2_)
        x3_ = self.conv3(x3_)
        x4_ = self.conv4(x4_)

        return self.sigmoid(x1_)*x1,self.sigmoid(x2_)*x2,self.sigmoid(x3_)*x3,self.sigmoid(x4_)*x4
