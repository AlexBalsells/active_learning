import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


def build_unet2d(input_layer,final_activation='sigmoid',num_classes=None,seed=42):
    binary = True
    if num_classes != None:
        binary = False


    #set seed for kernel_initializer
    init = tf.keras.initializers.HeNormal(seed=seed)
    nfilter = 16
    # First Layer
    conv1 = Conv2D(nfilter, (3,3), activation='relu', padding='same',
                  kernel_initializer=init)(input_layer)
    conv1 = Conv2D(nfilter, (3,3), activation='relu',kernel_initializer=init,
                    padding='same')(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)
    
    
    # Second Layer
    conv2 = Conv2D(nfilter, (3,3), activation='relu',padding='same',
                   kernel_initializer=init)(pool1)
    conv2 = Conv2D(nfilter,(3,3),activation='relu',padding='same',
                  kernel_initializer=init)(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)
    
    
    # Third Layer
    conv3 = Conv2D(nfilter,(3,3),activation='relu',padding='same',
                  kernel_initializer=init)(pool2)
    conv3 = Conv2D(nfilter,(3,3),activation='relu',padding='same',
                  kernel_initializer=init)(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)
    
    # Fourth and final Layer - call it middle
    convm = Conv2D(nfilter,(3,3),activation='relu',padding='same',
                   kernel_initializer=init)(pool3)
    convm = Conv2D(nfilter,(3,3),activation='relu',padding='same',
                  kernel_initializer=init)(convm)
    
    #Third Layer
    upsam3 = Conv2DTranspose(nfilter,(3,3),activation='relu',padding='same',
                             kernel_initializer=init,strides=(2,2))(convm)
    uconv3 = concatenate([upsam3,conv3])
    uconv3 = Conv2D(nfilter,(3,3),activation='relu',padding='same',
                   kernel_initializer=init)(uconv3)
    uconv3 = Conv2D(nfilter,(3,3),activation='relu',padding='same',
                   kernel_initializer=init)(uconv3)
    
    #Second Layer
    upsam2 = Conv2DTranspose(nfilter,(3,3),activation='relu',padding='same',
                   kernel_initializer=init,strides=(2,2))(uconv3)
    uconv2 = concatenate([upsam2,conv2])
    uconv2 = Conv2D(nfilter,(3,3),activation='relu',padding='same',
                   kernel_initializer=init)(uconv2)
    uconv2 = Conv2D(nfilter,(3,3),activation='relu',padding='same',
                   kernel_initializer=init)(uconv2)
    
    #First Layer
    upsam1 = Conv2DTranspose(nfilter,(3,3),activation='relu',padding='same',
                   kernel_initializer=init,strides=(2,2))(uconv2)
    uconv1 = concatenate([upsam1,conv1])
    uconv1 = Conv2D(nfilter,(3,3),activation='relu',padding='same',
                   kernel_initializer=init)(uconv1)
    uconv1 = Conv2D(nfilter,(3,3),activation='relu',padding='same',
                   kernel_initializer=init)(uconv1)
    
    #Output
    if binary:
        output_layer = Conv2D(1,(1,1),activation=final_activation,padding='same',
                              kernel_initializer=init)(uconv1)
    else:
        output_layer = Conv2D(num_classes,1,activation='softmax',padding='same',
                              kernel_initializer=init)(uconv1)
    Unet = Model(inputs=input_layer, outputs=output_layer)
    return Unet
