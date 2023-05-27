
from skimage import io
import numpy as np
import keras
import tensorflow as tf
import glob
import math
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,recall_score
from BFMD_SN_UNet_gci_cbam import *
from crop_to_shape_ import crop_to_shape





Imagelist_TEST = []
Masklist_TEST =  []

for image in sorted(glob.glob('/*')):
  image = io.imread(image)
  Imagelist_TEST.append(image)
for mask in sorted(glob.glob('/*')):
  mask = io.imread(mask)
  Masklist_TEST.append(mask)



desired_size = 1008
ImagesL=[]
LabelsL=[]
for i in range (0,len(Imagelist_TEST)):
    im = Imagelist_TEST[i]
    label = Masklist_TEST[i]
    old_size = im.shape[:2]  # old_size is in (height, width) format
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

ImagesTest = np.array(ImagesL)
LabelsTest = np.array(LabelsL)
ImagesTest = (ImagesTest.astype('float16') / 255.).astype('float16')
LabelsTest = (LabelsTest.astype('float16') / 255.).astype('float16')
LabelsTest =  np.expand_dims(LabelsTest, axis=3)


model = BFMD_SN_UNet_gci_cbam(input_shape=(1008,1008,3))
model.load_weights("")
# model = tf.keras.models.load_model("",compile=False)

y_test =  LabelsTest
y_test=crop_to_shape(y_test,(len(y_test), 960, 999, 1))

y_test1 = y_test

y_test = y_test.ravel()
y_test = y_test.astype(int)
y_pred = model.predict(ImagesTest)
y_pred= crop_to_shape(y_pred,(len(y_pred),960,999,1))

y_pred = y_pred.ravel()

y_pred = y_pred.astype(float)

y_pred_threshold =np.empty((y_pred.shape[0]))
threshold = 0.5

for i in range(y_pred.shape[0]):
    if y_pred[i]>=threshold:
        y_pred_threshold[i]=1
    else:
        y_pred_threshold[i]=0


tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()



Acc = accuracy_score(y_test, y_pred_threshold)
Auc =  roc_auc_score(y_test, list(np.ravel(y_pred)))
Sen = recall_score(y_test, y_pred_threshold)
Spe =  tn / (tn + fp)
Pre = tp/(tp+fp)
F1_SCORE = 2*tp/(2*tp+fn+fp)

print('Accuracy:',Acc)
print('Sensitivity:', Sen)
print('Specificity:', Spe)
print('AUC:', Auc)
print('Precision:',Pre)
print('Balanced Accuracy:', (Sen + Spe)/2)
print("F1:",F1_SCORE)



    
