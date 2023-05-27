# BFMD SN-U-net GCI-CBAM
**The open source code for the paper "Block Attention and Switchable Normalization based Deep Learning Framework for Segmentation of Retinal Vessels"**

_**Abstract:**_
The presence of high blood sugar levels damages blood vessels and causes an eye condition called diabetic retinopathy. The ophthalmologist can detect this disease by looking at the variations in retinal blood vasculature. Manual segmentation of vessels requires highly skilled specialists, and not possible for many patients to be done quickly in their daily routine. For these reasons, it is of great importance to isolate retinal vessels precisely, quickly, and accurately. The difficulty distinguishing the retinal vessels from the background, and the small number of samples in the databases make this segmentation problem difficult. In this study, we propose a novel network called Block Feature Map Distorted Switchable Normalization U-net with Global Context Informative Convolutional Block Attention Module (BFMD SN U-net with GCI- CBAM). We improve the traditional Fully Convolutional Segmentation Networks in multiple aspects with the proposed model as follows; The model converges in earlier epochs, adapts more flexibly to different data, is more robust against overfitting, and gets better feature refinement at different dilation rates to cope with different sizes of retinal vessels. We evaluate the proposed network on two reference retinal datasets, DRIVE and CHASE DB1, and achieve state-of-the-art performances with 97.00 % accuracy and 98.71 % AUC in DRIVE and 97.62 % accuracy and 99.11 % AUC on CHASE DB1 databases. Additionally, the convergence step of the model is reduced and it has fewer parameters than the baseline U-net. In summary, the proposed model surpasses the U-net -based approaches used for retinal vessel separation in the literature.

![image](https://github.com/sabrid369/BFMD-SN-U-net/assets/80791539/62b50b84-c365-44c6-8bd7-b6557fd3085b)

# BFMD SN U-net with GCI-CBAM
**We propose BFMD SN-Unet :**
_to overcome overfitting problem of the U-net,
overcome an important challenge of U-net design choice: Which Normalization Type to use. SN-U-net is able to learn which normalization is most useful automatically(Batch,Layer or Instance)_

**We propose a Global Context Informative Convolutional Block Attention Module :**
_to give attention where the information present on the spatil axis, or with other words suppress the noise in the image, and to learn different sizes of information more effectively as it has higher receptive field.(without increasing convolution parameters)_

_to learn a channel attention map by exploiting the inter-channel relationship of features and use Conv1d instead of MLP to improve computational efficiency_

# PyTorch Training 
You can start with data_augmentation_and_zero_padding.py and then train.py
# Pytorch Test
You can test with test.py

# Keras Training 
You can start with train.py
# Keras Test
You can test with test.py

# If you are inspired by our work, please cite this paper
@ARTICLE{10097749,
  author={Deari, Sabri and Oksuz, Ilkay and Ulukaya, Sezer},
  journal={IEEE Access}, 
  title={Block Attention and Switchable Normalization Based Deep Learning Framework for Segmentation of Retinal Vessels}, 
  year={2023},
  volume={11},
  number={},
  pages={38263-38274},
  doi={10.1109/ACCESS.2023.3265729}}
