import os
import cv2
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from skimage import io
from imutils import paths
from sklearn.model_selection import train_test_split

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

NUM_EPOCHS=200
BATCH_SIZE=4

DATASET_PATH = ""
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH)
GND_PATH = ""
GND_DATASET_PATH = os.path.join(GND_PATH)




imagePaths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))
gndPaths = sorted(list(paths.list_files(GND_DATASET_PATH)))



split = train_test_split(imagePaths, gndPaths,
	test_size=0.2, random_state=42)
(trainImages, testImages) = split[:2]
(trainGnd, valGnd) = split[2:]

transforms = transforms.Compose([
                                 transforms.ToPILImage(),transforms.ToTensor()])

trainDS = SegmentationDataset(imagePaths=trainImages, gndPaths=trainGnd,transforms=transforms)
valDS = SegmentationDataset(imagePaths=testImages, gndPaths=valGnd,transforms=transforms)

print(f" There is {len(trainDS)} examples in the training set...")
print(f" There is {len(valDS)} examples in the test set...")

trainLoader = DataLoader(trainDS, shuffle=True,batch_size=4,num_workers=os.cpu_count())
valLoader = DataLoader(valDS, shuffle=False,batch_size=4)


lossFunc = BCEWithLogitsLoss()
opt = Adam(model.parameters(), lr=0.001)
Train_Loss = []
Test_Loss=[]


model = BFMD_SN_UNet_with_GCI_CBAM(n_channels=3, n_classes=1)
model.to(device)


startTime = time.time()
for train_step in (range(NUM_EPOCHS)):
	# take the model to the training mode
	model.train()
	# initialize the total training&validation loss
	total_TrainLoss = 0
	total_TestLoss = 0
	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(device), y.to(device))
		# perform a forward pass and calculate the training loss
		pred = model(x)		
		y = y[:,:,0:510,0:510]
		loss = lossFunc(pred, y)
		# reset any previously accumulated gradients, then back-pro
		opt.zero_grad()
		loss.backward()
		opt.step() # optimizer
		total_TrainLoss += loss
	with torch.no_grad():
		model.eval()
		# loop over the validation set
		for (x, y) in valLoader:
			(x, y) = (x.to(device), y.to(device))
			pred = model(x)
			y = y[:,:,0:510,0:510]
			total_TestLoss += lossFunc(pred, y)
	avgTrainLoss = total_TrainLoss / trainSteps
	avgTestLoss = total_TestLoss / testSteps

	Train_Loss.append(avgTrainLoss.cpu().detach().numpy())
	Test_Loss.append(avgTestLoss.cpu().detach().numpy())
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))
endTime = time.time()
print("Total Time: {:.2f}s".format(endTime - startTime))

