import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D,Conv2DTranspose,concatenate


from switch_able_norm_2d import *
from block_feature_map_distortion import *

f=[16,32,64,128,256]
input_size=(592,592, 3) 

def BFMD_SN_UNet(f=16,kernel_size=(3,3),padding = 'same',strides = 1):

    x = Input(input_size)
    c1 = Conv2D(f, (3, 3), activation=None, padding="same")(x)
    c1 = BFMD(dist_prob=,block_size=)(c1)
    c1 = SwitchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(f, (3, 3), activation=None, padding="same")(c1)
    c1 = BFMD(dist_prob=,block_size=)(c1)
    c1 = SwitchNormalization()(c1)
    c1 = Activation('relu')(c1)



    p1 = MaxPooling2D((2,2),(2,2))(c1)

    c2 = Conv2D(f*2, (3, 3), activation=None, padding="same")(p1)
    c2 = BFMD(dist_prob=,block_size=)(c2)
    c2 = SwitchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(f*2, (3, 3), activation=None, padding="same")(c2)
    c2 = BFMD(dist_prob=,block_size=)(c2)
    c2 = SwitchNormalization()(c2)
    c2 = Activation('relu')(c2)

    

    p2 = MaxPooling2D((2,2),(2,2))(c2)

    c3 = Conv2D(f*4, (3, 3), activation=None, padding="same")(p2)
    c3 = BFMD(dist_prob=,block_size=)(c3)
    c3 = SwitchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(f*4, (3, 3), activation=None, padding="same")(c3)
    c3 = BFMD(dist_prob=,block_size=)(c3)
    c3 = SwitchNormalization()(c3)
    c3 = Activation('relu')(c3)



    p3 = MaxPooling2D((2,2),(2,2))(c3)

    cb = Conv2D(f*8, (3, 3), activation=None, padding="same")(p3)
    cb = BFMD(dist_prob=,block_size=)(cb)
    cb = SwitchNormalization()(cb)
    cb = Activation('relu')(cb)
    cb = Conv2D(f*8, (3, 3), activation=None, padding="same")(cb)
    cb = BFMD(dist_prob=,block_size=)(cb)
    cb = SwitchNormalization()(cb)
    cb = Activation('relu')(cb)
    

    dc3 = Conv2DTranspose(f*4, (3, 3), strides=(2, 2), padding="same")(cb)
    uc3 = concatenate([dc3, c3])


    uc3 = Conv2D(f*4, (3, 3), activation=None, padding="same")(uc3)
    uc3 = BFMD(dist_prob=,block_size=)(uc3)
    uc3 = SwitchNormalization()(uc3)
    uc3 = Activation('relu')(uc3)
    uc3 = Conv2D(f*4, (3, 3), activation=None, padding="same")(uc3)
    uc3 = BFMD(dist_prob=,block_size=)(uc3)
    uc3 = SwitchNormalization()(uc3)
    uc3 = Activation('relu')(uc3)


    dc2 = Conv2DTranspose(f*2, (3, 3), strides=(2, 2), padding="same")(uc3)
    uc2 = concatenate([dc2, c2])
    
    

    uc2 = Conv2D(f*2, (3, 3), activation=None, padding="same")(uc2)
    uc2 = BFMD(dist_prob=,block_size=)(uc2)
    uc2 = SwitchNormalization()(uc2)
    uc2 = Activation('relu')(uc2)
    uc2 = Conv2D(f*2, (3, 3), activation=None, padding="same")(uc2)
    uc2 = BFMD(dist_prob=,block_size=)(uc2)
    uc2 = SwitchNormalization()(uc2)
    uc2 = Activation('relu')(uc2)

    dc1 = Conv2DTranspose(f, (3, 3), strides=(2, 2), padding="same")(uc2)
    uc1 = concatenate([dc1, c1])
    

    uc1 = Conv2D(f, (3, 3), activation=None, padding="same")(uc1)
    uc1 = BFMD(dist_prob=,block_size=)(uc1)
    uc1 = SwitchNormalization()(uc1)
    uc1 = Activation('relu')(uc1)
    uc1 = Conv2D(f, (3, 3), activation=None, padding="same")(uc1)
    uc1 = BFMD(dist_prob=,block_size=)(uc1)
    uc1 = SwitchNormalization()(uc1)
    uc1 = Activation('relu')(uc1)
    
    output = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(uc1)

    model = Model(x,output)

    return model
