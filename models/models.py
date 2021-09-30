from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

def unet255(img_size, img_chan=1,  pretrained_weights = None):
    inputs = keras.Input(shape=img_size + (img_chan,))

    ### [First half of the network: downsampling inputs]
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    up6 = BatchNormalization()(up6)
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = BatchNormalization()(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation = sigmoid255)(conv9)    
    
    outputs = conv10
    # Define the model
    model = keras.Model(inputs, outputs)
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    
    return model

# Multi-Task SIMO network
def unet_mtl(img_size, img_chan = 1, pretrained_weights = None):
    inputs = keras.Input(shape=img_size + (img_chan,))

    ### [First half of the network: downsampling inputs] ###

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    sos_up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    sos_up6 = BatchNormalization()(sos_up6)
    sos_merge6 = concatenate([drop4,sos_up6], axis = 3)
    sos_conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_merge6)
    sos_conv6 = BatchNormalization()(sos_conv6)
    sos_conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_conv6)
    sos_conv6 = BatchNormalization()(sos_conv6)
    
    img_up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    img_up6 = BatchNormalization()(img_up6)
    img_merge6 = concatenate([drop4,img_up6], axis = 3)
    img_conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_merge6)
    img_conv6 = BatchNormalization()(img_conv6)
    img_conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_conv6)
    img_conv6 = BatchNormalization()(img_conv6)
    
    sos_up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(sos_conv6))
    sos_up7 = BatchNormalization()(sos_up7)
    sos_merge7 = concatenate([conv3,sos_up7], axis = 3)
    sos_conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_merge7)
    sos_conv7 = BatchNormalization()(sos_conv7)
    sos_conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_conv7)
    sos_conv7 = BatchNormalization()(sos_conv7)
    
    img_up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(img_conv6))
    img_up7 = BatchNormalization()(img_up7)
    img_merge7 = concatenate([conv3,img_up7], axis = 3)
    img_conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_merge7)
    img_conv7 = BatchNormalization()(img_conv7)
    img_conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_conv7)
    img_conv7 = BatchNormalization()(img_conv7)

    sos_up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(sos_conv7))
    sos_up8 = BatchNormalization()(sos_up8)
    sos_merge8 = concatenate([conv2,sos_up8], axis = 3)
    sos_conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_merge8)
    sos_conv8 = BatchNormalization()(sos_conv8)
    sos_conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_conv8)
    sos_conv8 = BatchNormalization()(sos_conv8)
    
    img_up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(img_conv7))
    img_up8 = BatchNormalization()(img_up8)
    img_merge8 = concatenate([conv2,img_up8], axis = 3)
    img_conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_merge8)
    img_conv8 = BatchNormalization()(img_conv8)
    img_conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_conv8)
    img_conv8 = BatchNormalization()(img_conv8)

    sos_up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(sos_conv8))
    sos_up9 = BatchNormalization()(sos_up9)
    sos_merge9 = concatenate([conv1,sos_up9], axis = 3)
    sos_conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_merge9)
    sos_conv9 = BatchNormalization()(sos_conv9)
    sos_conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_conv9)
    sos_conv9 = BatchNormalization()(sos_conv9)
    sos_conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_conv9)
    sos_conv9 = BatchNormalization()(sos_conv9)
    sos_conv10 = Conv2D(1, 1, name='sos_output')(sos_conv9)
    
    img_up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(img_conv8))
    img_up9 = BatchNormalization()(img_up9)
    img_merge9 = concatenate([conv1,img_up9], axis = 3)
    img_conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_merge9)
    img_conv9 = BatchNormalization()(img_conv9)
    img_conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_conv9)
    img_conv9 = BatchNormalization()(img_conv9)
    img_conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_conv9)
    img_conv9 = BatchNormalization()(img_conv9)
    img_conv10 = Conv2D(1, 1, name ='img_output')(img_conv9)
    
    outputs = [sos_conv10, img_conv10]
    # Define the model
    model = keras.Model(inputs, outputs)
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    
    return model

## Multi-Task MIMO network
def unet_mtl_mimo_deeper(img_size, img1_chan=1, img2_chan=3,  pretrained_weights = None):
    inputs_data = keras.Input(shape=img_size + (img1_chan,))
    inputs_uimg = keras.Input(shape=img_size + (img2_chan,))

    ### [First half of the network: downsampling inputs] ###
    
# Unfocused Image Encoding
    conv1_uimg =  Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs_uimg)
    conv1_uimg = BatchNormalization()(conv1_uimg)
    conv1_uimg =  Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_uimg)
    conv1_uimg = BatchNormalization()(conv1_uimg)
    pool1_uimg = MaxPooling2D(pool_size=(2, 2))(conv1_uimg)
    conv2_uimg = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1_uimg)
    conv2_uimg = BatchNormalization()(conv2_uimg)
    conv2_uimg = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_uimg)
    conv2_uimg = BatchNormalization()(conv2_uimg)
    pool2_uimg = MaxPooling2D(pool_size=(2, 2))(conv2_uimg)   
    conv3_uimg = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2_uimg)
    conv3_uimg = BatchNormalization()(conv3_uimg)
    conv3_uimg = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_uimg)
    conv3_uimg = BatchNormalization()(conv3_uimg)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs_data)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)   
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    
    conv3 = concatenate([conv3,conv3_uimg], axis = 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    sos_up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    sos_up6 = BatchNormalization()(sos_up6)
    sos_merge6 = concatenate([drop4,sos_up6], axis = 3)
    sos_conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_merge6)
    sos_conv6 = BatchNormalization()(sos_conv6)
    sos_conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_conv6)
    sos_conv6 = BatchNormalization()(sos_conv6)
    
    img_up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    img_up6 = BatchNormalization()(img_up6)
    img_merge6 = concatenate([drop4,img_up6], axis = 3)
    img_conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_merge6)
    img_conv6 = BatchNormalization()(img_conv6)
    img_conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_conv6)
    img_conv6 = BatchNormalization()(img_conv6)
    
    sos_up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(sos_conv6))
    sos_up7 = BatchNormalization()(sos_up7)
    sos_merge7 = concatenate([conv3,sos_up7], axis = 3)
    sos_conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_merge7)
    sos_conv7 = BatchNormalization()(sos_conv7)
    sos_conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_conv7)
    sos_conv7 = BatchNormalization()(sos_conv7)
    
    img_up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(img_conv6))
    img_up7 = BatchNormalization()(img_up7)
    img_merge7 = concatenate([conv3,img_up7], axis = 3)
    img_conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_merge7)
    img_conv7 = BatchNormalization()(img_conv7)
    img_conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_conv7)
    img_conv7 = BatchNormalization()(img_conv7)

    sos_up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(sos_conv7))
    sos_up8 = BatchNormalization()(sos_up8)
    sos_merge8 = concatenate([conv2,sos_up8], axis = 3)
    sos_conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_merge8)
    sos_conv8 = BatchNormalization()(sos_conv8)
    sos_conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_conv8)
    sos_conv8 = BatchNormalization()(sos_conv8)
    
    img_up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(img_conv7))
    img_up8 = BatchNormalization()(img_up8)
    img_merge8 = concatenate([conv2,img_up8], axis = 3)
    img_conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_merge8)
    img_conv8 = BatchNormalization()(img_conv8)
    img_conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_conv8)
    img_conv8 = BatchNormalization()(img_conv8)

    sos_up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(sos_conv8))
    sos_up9 = BatchNormalization()(sos_up9)
    sos_merge9 = concatenate([conv1,sos_up9], axis = 3)
    sos_conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_merge9)
    sos_conv9 = BatchNormalization()(sos_conv9)
    sos_conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_conv9)
    sos_conv9 = BatchNormalization()(sos_conv9)
    sos_conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sos_conv9)
    sos_conv9 = BatchNormalization()(sos_conv9)
    sos_conv10 = Conv2D(1, 1, name='sos_output')(sos_conv9)
    
    img_up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(img_conv8))
    img_up9 = BatchNormalization()(img_up9)
    img_merge9 = concatenate([conv1,img_up9], axis = 3)
    img_conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_merge9)
    img_conv9 = BatchNormalization()(img_conv9)
    img_conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_conv9)
    img_conv9 = BatchNormalization()(img_conv9)
    img_conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_conv9)
    img_conv9 = BatchNormalization()(img_conv9)
    img_conv10 = Conv2D(1, 1, name ='img_output')(img_conv9)
    
    inputs = [inputs_data, inputs_uimg]
    outputs = [sos_conv10, img_conv10]
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    
    return model


# Optional Custom activation function
def sigmoid255(x):
    return 255.0 * K.sigmoid(x)

# Plot loss function at completion of every epoch
class PlotProgress(keras.callbacks.Callback):
    
    def __init__(self, entity='loss'):
        self.entity = entity
        
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('{}'.format(self.entity)))
        self.val_losses.append(logs.get('val_{}'.format(self.entity)))
        self.i += 1
        
        clear_output(wait=True)
        plt.figure(0)
        plt.plot(self.x, self.losses, label="{}".format(self.entity))
        plt.plot(self.x, self.val_losses, label="val_{}".format(self.entity))
        bottom, top = plt.ylim()
        if np.mean(self.losses) > 10:
            bot = 1
        else:
            bot = 1e-9
        plt.ylim(bottom if bottom > 0 else bot, top if top < 1e7 else 1e7)
        plt.yscale('log')
        plt.legend()
        plt.show();
        
        plt.figure(1)
        plt.plot(self.x, self.losses, label="{}".format(self.entity))
        plt.plot(self.x, self.val_losses, label="val_{}".format(self.entity))
        bottom, top = plt.ylim()
        plt.ylim(bottom if bottom > 0 else 0, top if len(self.val_losses)<5 else max(self.val_losses[-5:]))
        plt.legend()
        plt.show();
