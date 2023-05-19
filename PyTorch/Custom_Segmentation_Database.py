from skimage import io
import numpy as np


class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, gndPaths, transforms):
		self.imagePaths = imagePaths
		self.gndPaths = gndPaths
		self.transforms = transforms
	def __len__(self):
		return len(self.imagePaths)
	def __getitem__(self, idx):
		imagePath = self.imagePaths[idx]
		image = io.imread(imagePath,plugin='pil')
		gnd = io.imread(self.gndPaths[idx], 0)
		gnd = np.expand_dims(gnd,2)
		if self.transforms is not None:
			image = self.transforms(image)
			gnd = self.transforms(gnd)
		# return a tuple of the image and its mask
		return (image, gnd)
