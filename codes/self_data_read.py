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



#%% positive augmentation 

# croping
# non-tensor 
def rand_crop_array(img): # input size (l,w, channel)
    return cv2.resize(img[100:400, 100:400,:], (100, 100))

# tensor 
im_size = (100, 100)
def rand_crop_tf(img):
    [temp1, temp2] =[randint(200,400),randint(200,400)]
    return tf.image.resize(tf.image.random_crop(img, size=(temp1, temp2, 40)), size = im_size).numpy()

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

i  = randint(40, 8900)
q1 =  np.transpose(data[i:i+40, :,:,1], [1,2,0]) # for now green channel only
k_pos1 = rand_crop_tf(q1)

i1 = i+np.random.choice([-1,1])+randint(10,20)
q2 =  np.transpose(data[i1:i1+40, :,:,1], [1,2,0]) # for now green channel only
k_pos2 = rand_crop_tf(q1)