#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:45:25 2021

@author: zahid
"""
# Important resource https://github.com/dragen1860/TensorFlow-2.x-Tutorials

from tensorflow.keras import Model, layers
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class ConvBNRelu(keras.Model):
    
    def __init__(self, ch, kernel_size=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        
        self.model = keras.models.Sequential([
            layers.Conv2D(ch, kernel_size =  kernel_size, strides=strides, padding=padding,
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        
        
    def call(self, x, training=None):
        
        x = self.model(x, training=training)
        
        return x 
    
# inception module

class InceptMod(keras.Model):
    def __init__(self, ch, strides = 1):
        super(InceptMod, self).__init__()
        
        self.ch = ch
        self.strides = strides
        
        self.conv1 = ConvBNRelu(ch, kernel_size=1, strides=strides)
        
        self.conv2 = ConvBNRelu(ch, kernel_size=3, strides=strides)
        
        self.conv3_1 = ConvBNRelu(ch, kernel_size=5,  strides=strides)
        
        self.conv3_2 = ConvBNRelu(ch, strides=strides)
        
        
        self.pool = keras.layers.MaxPooling2D(3, strides=1, padding='same')
        self.pool_conv = ConvBNRelu(ch, strides=strides)
        
        
    def call(self, x, training=None):
        x1 = self.conv1(x, training=training)

        x2 = self.conv2(x, training=training)
                
        x3_1 = self.conv3_1(x, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)
                
        x4 = self.pool(x)
        x4 = self.pool_conv(x4, training=training)
        
        # concat along axis=channel
        x = tf.concat([x1, x2, x3_2, x4], axis=3)
        
        return x
    

#%% Actual Network starts from Here. 

# Body network
class CNN_back(Model): # Vitamon network except inception layer
    # Set layers.
    def __init__(self):
        super(CNN_back, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = ConvBNRelu(32)

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = ConvBNRelu(64)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        
        self.conv3 = ConvBNRelu(64, kernel_size=3)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv4 = ConvBNRelu(96, kernel_size=3)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        
        self.incept1 = InceptMod(ch = 16, strides = 1)
        
        self.avgpool1 = layers.AveragePooling2D(2, strides= 2)
        
        self.incept2 = InceptMod(ch = 16, strides = 1)
        
        self.avgpool2 = layers.AveragePooling2D(2, strides= 2)
        
        self.flatten = layers.Flatten()


    # Set forward pass.
    def call(self, x, training=False):
        x = tf.reshape(x, [-1, 100, 100, 40])
        
        # x =self.concat1(xl)
        
        x = self.conv1(x, training=training)
        # print(x.shape)
        x = self.conv2(x, training=training)
        x = self.maxpool1(x)
        
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        
        x = self.maxpool2(x)
        
        x = self.incept1(x, training = training)
        
        x = self.avgpool1(x)
   
        x = self.incept2(x, training = training)
        
        x = self.avgpool2(x)
        # later added        
        # x = self.avgpool2(x)
        
        
        # later added        

        # print(x.shape)
        
        x = self.flatten(x)
        

        
        

        
        # print(x3.shape)

        return x



# Head network

class MtlNetwork_head(Model): # Vitamon network except inception layer
    # Set layers.
    def __init__(self, num_classes):
        super(MtlNetwork_head, self).__init__()
        
        self.fc1_1 = layers.Dense(512)
        
        self.fc1_2 = layers.Dense(128) # changed from 512
        self.out1 = layers.Dense(num_classes)
        self.act = tf.keras.layers.ReLU()

    # Set forward pass.
    def call(self, x, training=False):
        
        x = self.act(self.fc1_1(x))
        x = self.act(self.fc1_2(x))

        x = self.act(self.out1(x))
        # print(x3.shape)
        return x
    