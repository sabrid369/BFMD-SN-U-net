import keras
import tensorflow as tf
from keras.layers import multiply, Permute, Concatenate,Lambda,AveragePooling2D,MaxPooling2D,AveragePooling2D,GlobalAveragePooling2D,Add,Conv2D,Conv1D,GlobalMaxPooling2D,Reshape
from keras import backend as K


def unsqueeze(input):
    return K.expand_dims(input,axis=-1)

def squeeze(input):
    return K.squeeze(input,axis=-1)

shared_conv_1 = Conv1D(filters=1,kernel_size=3,strides=1,dilation_rate=1,padding="same")
shared_conv_2 = Conv1D(filters=1,kernel_size=3,strides=1,dilation_rate=2,padding="same")
shared_conv_3 = Conv1D(filters=1,kernel_size=3,strides=1,dilation_rate=3,padding="same",)
shared_conv_4 = Conv1D(filters=1,kernel_size=3,strides=1,dilation_rate=4,padding="same")


def GCI_CBAM(input):

    channel = input.shape[-1]

    avg_pool = GlobalAveragePooling2D()(input) # global avg pool of features
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = Permute((3, 1, 2))(avg_pool)
    avg_pool = Lambda(squeeze)(avg_pool)
    max_pool = GlobalMaxPooling2D()(input) # global max pool of features
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = Permute((3, 1, 2))(max_pool)
    max_pool = Lambda(squeeze)(max_pool)

    # First vein of GCI CBAM
    # channel attention
    avg_pool1 = shared_conv_1(avg_pool)
    avg_pool1 = Lambda(unsqueeze)(avg_pool1)
    avg_pool1 = Permute((2, 3, 1))(avg_pool1)
    max_pool1 = shared_conv_1(max_pool)
    max_pool1 = Lambda(unsqueeze)(max_pool1)
    max_pool1 = Permute((2, 3, 1))(max_pool1)
    channel_feature1 = Add()([avg_pool1, max_pool1])
    channel_feature1 = Activation('sigmoid')(channel_feature1)
    channel_feature1 = multiply([input,channel_feature1])    
    # spatial attention
    avg_pool1 = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_feature1)
    max_pool1 = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_feature1)
    concat1 = Concatenate(axis=3)([avg_pool1, max_pool1])  
    spatial_feature1 = Conv2D(filters=1,kernel_size=(7,7),strides=1,dilation_rate=(1,1),padding='same',activation='sigmoid',kernel_initializer = 'he_normal',use_bias=False)(concat1)
    # cbam feature
    cbam_feature1 = multiply([channel_feature1,spatial_feature1])


    # Second vein of GCI CBAM
    # channel attention
    avg_pool2 = shared_conv_2(avg_pool)
    avg_pool2 = Lambda(unsqueeze)(avg_pool2)
    avg_pool2 = Permute((2, 3, 1))(avg_pool2)
    max_pool2 = shared_conv_2(max_pool)
    max_pool2 = Lambda(unsqueeze)(max_pool2)
    max_pool2 = Permute((2, 3, 1))(max_pool2)
    channel_feature2 = Add()([avg_pool2, max_pool2])
    channel_feature2 = Activation('sigmoid')(channel_feature2)
    channel_feature2 = multiply([input,channel_feature2])    
    # spatial attention
    avg_pool2 = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_feature2)
    max_pool2 = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_feature2)
    concat2 = Concatenate(axis=3)([avg_pool2, max_pool2])  
    spatial_feature2 = Conv2D(filters=1,kernel_size=(7,7),strides=1,dilation_rate=(2,2),padding='same',activation='sigmoid',kernel_initializer = 'he_normal',use_bias=False)(concat2)
    # cbam feature
    cbam_feature2 = multiply([channel_feature2,spatial_feature2])
 

 
    # Third vein of GCI CBAM
    # channel attention
    avg_pool3 = shared_conv_3(avg_pool)
    avg_pool3 = Lambda(unsqueeze)(avg_pool3)
    avg_pool3 = Permute((2, 3, 1))(avg_pool3)
    max_pool3 = shared_conv_3(max_pool)
    max_pool3 = Lambda(unsqueeze)(max_pool3)
    max_pool3 = Permute((2, 3, 1))(max_pool3)
    channel_feature3 = Add()([avg_pool3, max_pool3])
    channel_feature3 = Activation('sigmoid')(channel_feature3)
    channel_feature3 = multiply([input,channel_feature3])    
    # spatial attention
    avg_pool3 = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_feature3)
    max_pool3 = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_feature3)
    concat3 = Concatenate(axis=3)([avg_pool3, max_pool3])  
    spatial_feature3 = Conv2D(filters=1,kernel_size=(7,7),strides=1,dilation_rate=(3,3),padding='same',activation='sigmoid',kernel_initializer = 'he_normal',use_bias=False)(concat3)
    # cbam feature
    cbam_feature3 = multiply([channel_feature3,spatial_feature3])
 

    # Fourth vein of GCI CBAM
    # channel attention
    avg_pool4 = shared_conv_4(avg_pool)
    avg_pool4 = Lambda(unsqueeze)(avg_pool4)
    avg_pool4 = Permute((2, 3, 1))(avg_pool4)
    max_pool4 = shared_conv_4(max_pool)
    max_pool4 = Lambda(unsqueeze)(max_pool4)
    max_pool4 = Permute((2, 3, 1))(max_pool4)
    channel_feature4 = Add()([avg_pool4, max_pool4])
    channel_feature4 = Activation('sigmoid')(channel_feature4)
    channel_feature4 = multiply([input,channel_feature4])    
    # spatial attention
    avg_pool4 = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_feature4)
    max_pool4 = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_feature4)
    concat4 = Concatenate(axis=3)([avg_pool4, max_pool4])  
    spatial_feature4 = Conv2D(filters=1,kernel_size=(7,7),strides=1,dilation_rate=(4,4),padding='same',activation='sigmoid',kernel_initializer = 'he_normal',use_bias=False)(concat4)
    # cbam feature
    cbam_feature4 = multiply([channel_feature4,spatial_feature4])

    return (Concatenate(axis=-1)([cbam_feature1,cbam_feature2,cbam_feature3,cbam_feature4]))
