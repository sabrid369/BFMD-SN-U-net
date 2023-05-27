
from skimage import io
import numpy as np
import keras
import tensorflow as tf
import glob
from sklearn.model_selection import train_test_split
import os
import random

from BFMD_SN_UNet_gci_cbam import *


Imagelist=[]
Labellist=[]
for image in sorted(glob.glob('/*')):
  image = io.imread(image)
  Imagelist.append(image)
for mask in sorted(glob.glob('/*')):
  mask = io.imread(mask)
  Labellist.append(mask)


desired_size = 592

ImagesL=[]
LabelsL = []

for i in range (0,len(Imagelist)):
    im = Imagelist[i]
    label = Labellist[i]
    old_size = im.shape[:2] 
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    ImagesL.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    LabelsL.append(temp)


Images = np.array(ImagesL)
Labels = np.array(LabelsL)
Images = (Images.astype('float16') / 255.).astype('float16')
Labels = (Labels.astype('float16') / 255.).astype('float16')
Labels =  np.expand_dims(Labels, axis=3)

X_Train , X_Test , Y_Train , Y_Test = train_test_split(Images,Labels,test_size = 0.1 , random_state = 369,shuffle=True)

model = BFMD_SN_UNet_gci_cbam(input_shape=(592,592,3))
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
checkpoint = keras.callbacks.ModelCheckpoint('_{epoch:04d}.h5',save_best_only=True,monitor="val_accuracy") 


history = model.fit(X_Train, Y_Train, epochs= 200 , batch_size=2,validation_data=(X_Test,Y_Test),callbacks=[checkpoint])












