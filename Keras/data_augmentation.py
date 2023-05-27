import numpy as np
import os
from skimage import io
import glob
import imgaug as ia
import imgaug.augmenters as iaa
import random
from PIL import Image


Imagelist=[]
Labellist=[]
for image in sorted(glob.glob('/*')):
  image = io.imread(image)
  Imagelist.append(image)
for mask in sorted(glob.glob('/*')):
  mask = io.imread(mask)
  Labellist.append(mask)

# Rotate by 90,180,270
x=[90,180,270]
for ii in range(0,len(x)):
  for idx in range(0,20):
    x1 = x[ii]
    image = Imagelist[idx]
    rotate=iaa.Affine(rotate=x1,mode='constant',deterministic=True)
    rotated_image=rotate.augment_image(image)
    rotated_image = np.array(rotated_image)
    Imagelist.append(rotated_image) 
    mask = Labellist[idx]
    rotate=iaa.Affine(rotate=x1,mode='constant',deterministic=True)
    rotated_mask=rotate.augment_image(mask)
    Labellist.append(rotated_mask)

# Random rotation
for i in range(0,1):
  for idx in range(0,20):
    x1 = random.uniform(0,359)
    image = Imagelist[idx]
    rotate=iaa.Affine(rotate=x1,mode='constant')
    rotated_image=rotate.augment_image(image)
    Imagelist.append(rotated_image) 
    mask = Labellist[idx]
    rotate=iaa.Affine(rotate=x1,mode='constant')
    rotated_mask=rotate.augment_image(mask)
    Labellist.append(rotated_mask)

# random crop
for i in range(0,1):
  for idx in range(0,20):
      x = random.uniform(0.10,0.15)
      image = Imagelist[idx]
      crop = iaa.Crop(percent=(x, x)) # crop image
      corp_image=crop.augment_image(image)
      Imagelist.append(corp_image)
      mask = Labellist[idx]
      crop = iaa.Crop(percent=(x, x)) # crop mask
      corp_mask=crop.augment_image(mask)
      Labellist.append(corp_mask)


# SHEAR
for i in range(0,1):
  for idx in range(0,20):
    x1 = random.randint(-8,8)
    image = Imagelist[idx]
    shearr = iaa.Affine(shear=x1)
    shear_image = shearr.augment_image(image)
    Imagelist.append(shear_image)
    mask = Labellist[idx]
    shear_mask = shearr.augment_image(mask)
    Labellist.append(shear_mask)



# gamma correction
for idx in range(0,20):
  aug = iaa.GammaContrast((2.0), per_channel=True)
  im = Imagelist[idx]
  im = aug.augment_image(im)
  Imagelist.append(im)
  mask = Labellist[idx]
  Labellist.append(mask)



# linear contrast
for idx in range(0,20):
  aug = iaa.LinearContrast((1.4))
  im = Imagelist[idx]
  im = aug.augment_image(im)
  Imagelist.append(im)
  mask = Labellist[idx]
  Labellist.append(mask)


# gaussian noise
for idx in range(0,20):
  x1 = random.uniform(0.01,0.02)
  im = Imagelist[idx]
  aug = iaa.AdditiveGaussianNoise(scale=x1*255)
  im = aug.augment_image(im)
  Imagelist.append(im)
  mask = Labellist[idx]
  Labellist.append(mask)

# brightening
from PIL import Image, ImageEnhance
for image in sorted(glob.glob('/*')):
  c1 = random.uniform(1.5,2)
  image = Image.open(image)
  enhancer = ImageEnhance.Brightness(image)
  im_output = enhancer.enhance(c1)
  im = np.array(im_output)
  Imagelist.append(im)
for mask in sorted(glob.glob('/*')):
  mask = io.imread(mask)
  Labellist.append(mask)

for idx in range(0,len(Imagelist):
    img = Imagelist[idx]
    label = Labellist[idx]
                 
    cv2.imwrite("filename"+str(idx)+'.png', img)
    cv2.imwrite("filename"+str(idx)+'.png', label)





                 
  
