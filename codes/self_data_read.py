#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:06:57 2020

@author: zahid
"""
#%% libraries

import tensorflow as tf

import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

import numpy as np

import cv2

import glob

from scipy.io import loadmat

import random

from random import seed, randint

from sklearn.model_selection import train_test_split

import pandas as pd


#%%  Data Load Parts




# load Pathdir
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/IR'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_raw'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_demosaiced'

path_dir = '../../../Dataset/Personal_collection/sub2_emon/col3/'

ppgtotal =  pd.read_csv(path_dir +'Emon_lab/BVP.csv')
EventMark = pd.read_csv(path_dir+'Emon_lab/tags.csv')

dataPath = os.path.join(path_dir, '*.MOV')

files = glob.glob(dataPath)  # care about the serialization
# end load pathdir
list.sort(files) # serialing the data

# Take time stamp and multiple by 64. Take starting time of the BVP file, 
# subtract the tags.csv from the BVP start time, multiply by 64 to get the sample number. 

#%% Load Video and load Mat file

# find start position by pressing the key position in empatica
# test1.MOV led appear at the 307th frame.

# perfect alignment! checked by (time_eM_last-time_eM_first)*30+start_press_sample  should
# give the end press in video



data = []
im_size = (500,500)

cap = cv2.VideoCapture(files[0])

import pdb

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret==False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
   # gray  = gray[:,:,:]
    gray =  gray[:900, 600:1500,:]
   
    gray = cv2.resize(gray, im_size)
    

   
    data.append(gray)
    
    # pdb.set_trace()
    
    #cv2.imshow('frame', gray)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


fps = cap.get(cv2.CAP_PROP_FPS)
    
cap.release()
cv2.destroyAllWindows()
data =  np.array(data)
#print(data.nbytes)

#%% data clearning 
data =data[1000:10000]

#%% positive augmentation 

# croping
# # non-tensor 
# def rand_crop_array(img): # input size (l,w, channel)
#     return cv2.resize(img[100:400, 100:400,:], (100, 100))

# tensor 
im_size = (100, 100)
def rand_crop_tf(img):
    [temp1, temp2] =[randint(200,400),randint(200,400)]
    return tf.image.resize(tf.image.random_crop(img, size=(temp1, temp2, 40)), size = im_size)

#%% negative augmentation 
from random import randint
# we can use multiple shuffling
def rand_frame_shuf(img):
    temp = np.arange(40)
    np.random.shuffle(temp)
    return img[:,:, temp]

def frame_repeat(img):
    temp = randint(0,39)
    img[:,:,0:40] = img[:,:, temp:temp+1]
    return img
    

# Use of np.stack 



#%% Data preparation
import pdb

def get_data():
    i  = randint(40, 8900)
    q1 =  np.transpose(data[i:i+40, :,:,1], [1,2,0]) # for now green channel only
    k_pos1 = rand_crop_tf(q1)
    k_pos11 = tf.image.resize(np.transpose(data[(i+26):i+66, :,:,1], [1,2,0]), size = im_size)
    ll = list([k_pos1, k_pos11])
    
    # pdb.set_trace()
    
    k_pos = ll[randint(0, len(ll)-1)]    
    
    i1 = i+np.random.choice([-1,1])*randint(13,18)
    q2 =  np.transpose(data[i1:i1+40, :,:,1], [1,2,0]) # for now green channel only
    k_pos2 = rand_crop_tf(q2)
    
    q1 = tf.image.resize(q1, size = im_size)
    q2 = tf.image.resize(q2, size = im_size)
    
    k_neg1 = rand_frame_shuf(q1.numpy())
    k_neg2 = frame_repeat(q1.numpy())
    
    k_neg3 = rand_frame_shuf(q2.numpy())
    k_neg5 = rand_frame_shuf(q2.numpy())
    k_neg4 = frame_repeat(q2.numpy())
    
    x_train = tf.stack([q1, k_pos, q2, k_pos2, k_neg1, k_neg2, k_neg3, k_neg4, k_neg5])/255.0
    return x_train

# simclr configuration later
# good version of code https://stackoverflow.com/questions/62793043/tensorflow-implementation-of-nt-xent-contrastive-loss-function
# simclr loss https://github.com/margokhokhlova/NT_Xent_loss_tensorflow

#%% loss function 

class Contrastive_loss(tf.keras.layers.Layer):
    def __init__(self, tau = 1, **kwargs):
        super(Contrastive_loss, self).__init__()
        self.tau = tau
        self.similarity = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)
        self.criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        
    def get_config(self):
        return {"tau": self.tau}
    
    def call(self, embed, embeds):
        mul_res = tf.exp(-1*self.similarity(embed, embeds)/self.tau)
        logits = mul_res/tf.math.reduce_sum(mul_res)
        y = tf.one_hot(0, depth=mul_res.shape[0])
        return self.criterion(y, logits)

# did not work straight forwardly!! may need to find way around

# triplet/max margin loss

#%% Network Definition 

from net_work_def import CNN_back, MtlNetwork_head
#%% 
body = CNN_back()
proj_head = MtlNetwork_head(20)
neural_net = tf.keras.Sequential([body, proj_head])


#%% Optimization 

loss_crit = Contrastive_loss(0.25)

optimizer= tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

optimizer1  = tf.optimizers.SGD(learning_rate=0.0004)


def trainNetProb(net):
    for step in range(100000):
        x = get_data()
        with tf.GradientTape() as g:
            pred =  net(x[0], training = True) 
            preds = net(x[1:], training = True)
            loss =  loss_crit(pred,preds)  
        trainable_variables =  net.trainable_variables
        gradients =  g.gradient(loss, trainable_variables)  
        optimizer1.apply_gradients(zip(gradients, trainable_variables))
        
        if step % (500) == 0:
            print("step %i, loss: %f" %(step, loss))
        
#%% Training main
# importance of the sensitivity

# training true and false matters really!!

with tf.device('gpu:0'): #very important
    trainNetProb(neural_net)
