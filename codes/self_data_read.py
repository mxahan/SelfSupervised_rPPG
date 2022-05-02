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

# from sklearn.model_selection import train_test_split

import pandas as pd


#%%  Data Load files from the directory

# Select the source file [either MERL or Collected Dataset or ]


# load Pathdir
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/IR'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_raw'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_demosaiced'

path_dir = '../../../Dataset/Personal_collection/MPSC_rppg/subject_003/trial_002/video/'



dataPath = os.path.join(path_dir, '*.MOV')

files = glob.glob(dataPath)  # care about the serialization
# end load pathdir
list.sort(files) # serialing the data

# Take time stamp and multiple by 64. Take starting time of the BVP file, 
# subtract the tags.csv from the BVP start time, multiply by 64 to get the sample number. 


#%% Load the Video and corresponding GT Mat file

# find start position by pressing the key position in empatica
# perfect alignment! checked by (time_eM_last-time_eM_first)*30+start_press_sample  should
# give the end press in video


def data_read(files, im_size = (200, 200)):
    data = []
    cap = cv2.VideoCapture(files[0])
    
    # import pdb
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret==False:
            break
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        gray = gray[:,:,1]
        gray = gray[50:1000, 620:1380]
        gray = cv2.resize(gray, im_size)
        # pdb.set_trace()
        data.append(gray)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    # fps = cap.get(cv2.CAP_PROP_FPS)
        
    cap.release()
    cv2.destroyAllWindows()
    data =  np.array(data)
    
    return data

data = data_read(files, im_size = (200, 200) )


#%% PPG signal selection and alignment. 

# The starting points are the crucial, 
# this section needs select both the sratrting of video and the ppg point
# check fps and starting time in BVP.csv
# Match the lines from the supplimentary text file for the data
def alignment_data(data, path_dir):
    ppgtotal =  pd.read_csv(path_dir +'../empatica_e4/BVP.csv')
    EventMark = pd.read_csv(path_dir+'../empatica_e4/tags.csv')
    evmarknp =  EventMark.to_numpy()
    ppgnp =  ppgtotal.to_numpy()
    start_gap =  evmarknp[0] - 1594845887
    end_point =  evmarknp[1] - evmarknp[0]
    
    ppgnp_align =  ppgnp[np.int(start_gap*64):np.int((start_gap+end_point)*64)]
    
    data_align = data[1110:1110+np.int(end_point*30)+5]
    ppgnp_align = np.reshape(ppgnp_align, [ppgnp_align.shape[0],1]) 

    return data_align, ppgnp_align

data_align, ppgnp_align = alignment_data(data, path_dir) 
del data

#%% pickle save
import pickle
input("pickle save ahead")
# Saving 
save_path  =  '../../../Dataset/Personal_collection/MPSC_rppg/Pickle_files_rppg/sub_002_003.pkl'
def pickle_save(data_align, ppgnp_align, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump([data_align, ppgnp_align], f)  #NickName: SAD

# pickle_save(data_align, ppgnp_align, save_path)

#%% pickle load
# Loading 
input("pickle load ahead")

def load_pickle(save_path):
    with open(save_path, 'rb') as f:
        a,b = pickle.load(f)
        return a, b

data_align, ppgnp_align= load_pickle(save_path= save_path)


#%% Prepare data loader
import numpy 

from random import choice
class data_loader():
    def __init__(self, video, ppg= None, im_size = (100, 100), frame_cons = 40, bs = 1):
        self.video = video
        self.max_len = video.shape[0] - 200
        self.ppg =  ppg
        self.im_size = im_size
        self.frame_cons = frame_cons
        self.bs = bs
    
    def get_sup_samp(self, pos  = None, frame_gap = 1, reSiz = True):
        
        if pos==None:
            pos = randint(50, self.max_len)
        img = np.transpose(self.video[pos:pos+self.frame_cons*frame_gap:frame_gap,:,:],[1,2,0])
        p_point = np.int(np.round(pos*64/30))
        ppg_gt = self.ppg[p_point: p_point+85*frame_gap: frame_gap, 0]
        
        ppg_gt = ppg_gt-np.min(ppg_gt)
        
        ppg_gt = (ppg_gt/np.max(ppg_gt))*2 -1
        
        if reSiz:
            img =  self.img_resize(img)
        
        return img, ppg_gt, pos
    
    def get_sup_data(self, bs = None):
        vs, gt = [], []
        if bs ==None: bs =self.bs
        for i in range(bs):
            frame_gap= 1 if random.random()>0.90 else 2
            sas, gtss, _ =  self.get_sup_samp(frame_gap=frame_gap)
            vs.append(sas); gt.append(gtss)
        return tf.cast(tf.stack(vs), tf.float32)/255.0, tf.cast(tf.stack(gt), tf.float32)
    
    def rand_frame_shuf(self, img):
        img = img.numpy()
        temp = np.arange(40)
        np.random.shuffle(temp)
        return img[:,:, temp]
    
    def frame_repeat(self, img):
        img = img.numpy()
        temp = randint(0,39)
        img[:,:,0:40] = img[:,:, temp:temp+1]
        return img
    
    def img_resize(self, img):
        return tf.image.resize(img, self.im_size)
    
    def rand_crop_tf(self, pos):
        img, _, _ =  self.get_sup_samp(pos= pos, reSiz=False)
        [temp1, temp2] =[randint(120,180),randint(120,180)]
        img = tf.image.random_crop(img, size=(temp1, temp2, 40))
        
        return self.img_resize(img)
    
    def fps_halfing(self, pos):
        img, ppg_gt, _ = self.get_sup_samp(pos= pos, frame_gap= 2)
        return img, ppg_gt
    
    def img_shifted(self, pos, sv):
        img, _, _ =  self.get_sup_samp(pos = pos+sv)   
        return img
    
    def get_CL_data(self, pve = 1, nve = 5):
        vs = []
        query, _, pos = self.get_sup_samp()
        vs.append(query)
        pos_op = [0, 1]
        for _ in range(pve):
            dum = choice(pos_op)
            if dum ==0:
                pq = self.img_shifted(pos = pos, sv = choice(list(range(19,23)))*choice([-1, 1]))
                vs.append(pq)
            elif dum ==1:
                pq = self.rand_crop_tf(pos= pos)
                vs.append(pq)
        
        neg_op = [0, 1, 2, 3]
        for _ in range(nve):
            dum = choice(neg_op)
            if dum==0:
                nq = self.img_shifted(pos = pos, sv = choice(list(range(9,12)))*choice([-1, 1]))
                vs.append(nq)
            elif dum==1:
                nq = self.frame_repeat(query)
                vs.append(nq)
            elif dum==2:
                nq = self.rand_frame_shuf(query)
                vs.append(nq)
            elif dum==3:
                nq, _ = self.fps_halfing(pos = pos)
                vs.append(nq)
                
        return  tf.cast(tf.stack(vs), tf.float32)/255.0
 
            




# Use of np.stack 





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
        # y = tf.one_hot(0, depth=mul_res.shape[0])
        # return self.criterion(y, logits)
        return -tf.math.log(logits[0])


# triplet/max margin loss

#%% Network Definition 

from net_work_def import CNN_back, MtlNetwork_head
#%% 
body = CNN_back()
proj_head = MtlNetwork_head(64)
neural_net = tf.keras.Sequential([body, proj_head])


#%% Optimization 

loss_crit = Contrastive_loss(0.05)

optimizer= tf.keras.optimizers.Adam(learning_rate=0.0008, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

optimizer1  = tf.optimizers.SGD(learning_rate=0.001)


def trainNetProb(net):
    samp_load =  data_loader(video=data_align, ppg = ppgnp_align, bs = 1)
    for step in range(40000):
        x = samp_load.get_CL_data(pve = 1, nve = 5)
        with tf.GradientTape() as g:
            pred =  net(x, training = True) 
            loss =  loss_crit(pred[0:1],pred[1:])  
        trainable_variables =  net.trainable_variables
        gradients =  g.gradient(loss, trainable_variables)  
        optimizer1.apply_gradients(zip(gradients, trainable_variables))
        
        if step % (500) == 0:
            print("step %i, loss: %f" %(step, loss))
        
#%% Training main

with tf.device('gpu:0'): #very important
    trainNetProb(neural_net)