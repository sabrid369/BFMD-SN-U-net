import os
import cv2
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from skimage import io
from imutils import paths
import torch
import torch.nn as nn
import torch.nn.functional as F


from block_feature_map_distortion import *
from switchable_norm_2d import *
from bfmd_sn_u_net_gci_cbam_blocks import *
from GCI_CBAM import *

from Custom_Segmentation_Database import SegmentationDataset
from BFMD_SN_U_net_GCI_CBAM import BFMD_SN_UNet_with_GCI_CBAM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

imagePaths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))
gndPaths = sorted(list(paths.list_files(GND_DATASET_PATH)))


testDS = SegmentationDataset(imagePaths=imagePaths, gndPaths=gndPaths,transforms=transforms)
print(f" There is {len(testDS)} examples in the training set...")

testLoader = DataLoader(testDS, shuffle=False,batch_size=4,num_workers=os.cpu_count())

lossFunc = BCEWithLogitsLoss()
Test_Loss=[]


model = BFMD_SN_UNet_with_GCI_CBAM(n_channels=3, n_classes=1)
model.load_state_dict(torch.load(""))
model.to(device)
model.eval()

total_TestLoss = 0
for (x, y) in testLoader:
    with torch.no_grad():
        (x, y) = (x.to(device), y.to(device))
        pred = model(x)
        pred = pred.detach().cpu().np()
        y =  y.detach().cpu().np()

        pred = crop_to_shape(pred,(len(pred),584,565,1))
        y = crop_to_shape(y,(len(y),584,565,1))
        total_TestLoss += lossFunc(pred, y)


print("Test loss: {:.6f}".format(total_TestLoss))    





