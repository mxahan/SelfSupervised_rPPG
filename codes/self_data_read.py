#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Apr 23 10:06:57 2020

@author: zahid
"""


#%% librariesa

import tensorflow as tf

import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

import numpy as np
# import cv2

import glob

from scipy.io import loadmat

import random

from random import seed, randint

# from sklearn.model_selection import train_test_split

import pandas as pd
import pickle

#%%  Data Load files from the directory

# Select the source file [either MERL or Collected Dataset or ]


# load Pathdir
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/IR'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_raw'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_demosaiced'

path_dir = '../../../Dataset/Personal_collection/MPSC_rppg/subject_006/trial_001/video/'



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
        gray = gray[0:1000, 400:1400]
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

# data = data_read(files, im_size = (200, 200) )


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
    start_gap =  evmarknp[0] -   1599770587
    end_point =  evmarknp[1] - evmarknp[0]
    
    ppgnp_align =  ppgnp[np.int(start_gap*64):np.int((start_gap+end_point)*64)]
    
    data_align =  data[676 : 676 +np.int(end_point*30)]
    
    ppgnp_align = np.reshape(ppgnp_align, [ppgnp_align.shape[0],1]) 

    return data_align, ppgnp_align

data_align, ppgnp_align = alignment_data(data, path_dir) 
del data

#%% pickle save

input("pickle save ahead")
# Saving 
save_path  =  '../../../Dataset/Personal_collection/MPSC_rppg/Pickle_files_rppg/sub_006_001.pkl'
def pickle_save(data_align, ppgnp_align, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump([data_align, ppgnp_align], f)  #NickName: SAD

# pickle_save(data_align, ppgnp_align, save_path)

#%% pickle load
# Loading 
def get_pkl_files():
    save_path  =  '../../../Dataset/Personal_collection/MPSC_rppg/Pickle_files_rppg/'
    
    dataPath = os.path.join(save_path, '*.pkl')
    
    files = glob.glob(dataPath)  # care about the serialization
    # end load pathdir
    list.sort(files) # serialing the data
    return files 


input("pickle load ahead")

def load_pickle(save_path):
    with open(save_path, 'rb') as f:
        a,b = pickle.load(f)
        return a, b


def get_data(files):
    for i in [2]:
        data_align, ppgnp_align= load_pickle(save_path= files[i])
        if 'data_a' in locals():
            data_a = np.append(data_a, data_align, axis = 0)
            ppg_a = np.append(ppg_a,ppgnp_align, axis =0)
        else:
            data_a = data_align
            ppg_a = ppgnp_align
        
    return data_a, ppg_a



files = get_pkl_files()
data_align, ppgnp_align =  get_data(files)



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
        
        
        
        if self.ppg.any() != None:
            p_point = np.int(np.round(pos*64/30))
            ppg_gt = self.ppg[p_point: p_point+85*frame_gap: frame_gap, 0]
            
            ppg_gt = ppg_gt-np.min(ppg_gt)
            
            ppg_gt = (ppg_gt/np.max(ppg_gt))*2 -1
        else: 
            ppg_gt = None
            
        if reSiz:
            img =  self.img_resize(img)
        
        return img, ppg_gt, pos
    
    def get_sup_data(self, bs = None):
        vs, gt = [], []
        if bs ==None: bs =self.bs
        for i in range(bs):
            frame_gap= 1 if random.random()<0.90 else 2
            sas, gtss, _ =  self.get_sup_samp(frame_gap=frame_gap)
            vs.append(sas); gt.append(gtss)
        return tf.cast(tf.stack(vs), tf.float32)/255.0, tf.cast(tf.stack(gt), tf.float32)
    
    def get_pair_supD(self):
        vs, gt = [], []
        pos = randint(40, self.max_len)
        sas, gtss, _ =  self.get_sup_samp(pos = pos, frame_gap=1)
        vs.append(sas); gt.append(gtss)
        
        frame_gap= 1 if random.random()<0.90 else 2
        
        if frame_gap==2:
            s_ = 0
            sas, gtss, _ =  self.get_sup_samp(pos = pos, frame_gap=frame_gap)
        else:
            s_ = randint(5,25)
            sas, gtss, _ =  self.get_sup_samp(pos = pos+s_, frame_gap=frame_gap)
        vs.append(sas); gt.append(gtss)
        
        return tf.cast(tf.stack(vs), tf.float32)/255.0, tf.cast(tf.stack(gt), tf.float32), s_
        
        
        
    
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
    
    def rand_crop_tf(self, pos, frame_gap = 1):
        img, _, _ =  self.get_sup_samp(pos= pos, frame_gap=frame_gap, reSiz=False)
        [temp1, temp2] =[randint(120,180),randint(120,180)]
        img = tf.image.random_crop(img, size=(temp1, temp2, 40))
        
        return self.img_resize(img)
    
    def fps_halfing(self, pos):
        img, ppg_gt, _ = self.get_sup_samp(pos= pos, frame_gap= 2)
        return img, ppg_gt
    
    def img_shifted(self, pos, sv, frame_gap = 1):
        img, _, _ =  self.get_sup_samp(pos = pos+sv, frame_gap=frame_gap)   
        return img
    
    def pos_sample(self, pos, frame_gap = 1, pos_op = [0, 1]):
        dum = choice(pos_op)
        if dum ==0:
            pq = self.img_shifted(pos = pos, sv = choice(list(range(19,23)))*choice([-1, 1])
                                  , frame_gap=frame_gap)
        elif dum ==1:
            pq = self.rand_crop_tf(pos= pos, frame_gap=frame_gap)    
        return pq
    
    def neg_sample(self, query,pos, neg_op = [0,1,2,3]):
        dum = choice(neg_op)
        if dum==0:
            nq = self.img_shifted(pos = pos, sv = choice(list(range(9,12)))*choice([-1, 1]))
        elif dum==1:
            nq = self.frame_repeat(query)
        elif dum==2:
            nq = self.rand_frame_shuf(query)
        elif dum==3:
            nq, _ = self.fps_halfing(pos = None)

        return nq       

    def get_CL_data(self, pve = 1, nve = 5):
        vs = []
        frame_gap = 1 if random.random() <0.9 else 2
        query, _, pos = self.get_sup_samp(pos = None, frame_gap= frame_gap)
        vs.append(query)
        
        for _ in range(pve):
            pq = self.pos_sample(pos = pos, frame_gap=frame_gap)
            vs.append(pq)
        
        neg_op = [0,3] if frame_gap ==1 else [0,1,2]
        for _ in range(nve):
            nq = self.neg_sample(query = query, pos = pos, neg_op=neg_op)
            vs.append(nq)
        return  tf.cast(tf.stack(vs), tf.float32)/255.0
 
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
        logits = -1*self.similarity(embed, embeds)/self.tau
        # logits = logits/tf.math.reduce_sum(logits)
        y = tf.one_hot(0, depth=logits.shape[0])
        return self.criterion(y, logits)
        # return -tf.reduce_sum(tf.math.log(tf.nn.softmax(mul_res)[0]))


# triplet/max margin loss


#%% Discriminator losses
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return 0.5*cross_entropy(tf.ones_like(fake_output), fake_output)

def shifted_loss(op_1, op_2, s_):
    if s_>0:
        s_ = np.int(np.round(s_*64/30))
        losss=tf.keras.losses.MSE(y_true = op_1[:, s_:85], y_pred = op_2[:, 0:(85-s_)])
    else:
        losss=tf.keras.losses.MSE(y_true = op_1[:, 0:85:2], y_pred = op_2[:, 0:43])
    return losss

#%% Network Definition 

from net_work_def import CNN_back, MtlNetwork_head, PPG_discriminator, PPG_generator

#%% 

def get_network(head = 85):
    
    body = CNN_back()
    # proj_head = MtlNetwork_head(head)
    proj_head = PPG_generator()
    neural_net = tf.keras.Sequential([body, proj_head])
    
    return neural_net

neural_net =  get_network(85)

ppg_disc = PPG_discriminator()
ppg_gen = neural_net

# https://www.tensorflow.org/guide/keras/transfer_learning

#%% Supervised Loss 
def RootMeanSquareLoss1(y,x):
    # pdb.set_trace()  
    loss = tf.keras.losses.MSE(y_true = y, y_pred =x)  # initial one
    #return tf.reduce_mean(loss)  # some other shape similarity
     
    loss2 = tf.reduce_mean((tf.math.abs(tf.math.sign(y))-tf.math.sign(tf.math.multiply(x,y))),axis = -1)
    # print(loss2.shape)
    
    # print(tf.reduce_mean(loss), tf.reduce_mean(loss2))
    return tf.reduce_mean(loss + 0.5*loss2)


#%% Optimization 

# loss_crit = Contrastive_loss(tau=0.05)
loss_crit = Contrastive_loss(tau=0.05)

optimizer= tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

optimizer1  = tf.optimizers.SGD(learning_rate=0.0005)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

loss_v = []

# give a zero loss to the frame repeating


def wt_save(neural_net, save_mdl = '../../../Dataset/Merl_Tim/NNsave/SavedWM/Models/ssl_model_2'):
    neural_net.save_weights(save_mdl)

def trainNetProb(net):
    samp_load =  data_loader(video=data_align, ppg = ppgnp_align, bs = 1)
    for step in range(100000):
        if random.random() <0.9:
            x,y = samp_load.get_sup_data(bs = 8)
            with tf.GradientTape() as g:
                pred =  net(x, training = True) 
                loss =  RootMeanSquareLoss1(y,pred)  
        else:
            x,y, s_ = samp_load.get_pair_supD()
            with tf.GradientTape() as g:
                pred =  net(x, training = True) 
                loss =  shifted_loss(pred[0:1], pred[1:2], s_) 
            
        trainable_variables =  net.trainable_variables
        gradients =  g.gradient(loss, trainable_variables)  
        optimizer1.apply_gradients(zip(gradients, trainable_variables))
        
        if step % (500) == 0:
            loss_v.append(loss)
            print("step %i, loss: %f" %(step, loss))
            
        if step % (9999) == 0:
            wt_save(net, save_mdl = '../../../Dataset/Merl_Tim/NNsave/SavedWM/Models/ssl_mods_2')


def train_gen_model():
    samp_load =  data_loader(video=data_align, ppg = ppgnp_align, bs = 1)
    for step in range(3000):
        
        x = samp_load.get_CL_data(pve = 1, nve = 8)
        with tf.GradientTape() as g:
            pred =  neural_net(x, training = True) 
            loss =  loss_crit(pred[0:1],pred[1:])  
        trainable_variables =  neural_net.trainable_variables
        gradients =  g.gradient(loss, trainable_variables)  
        optimizer1.apply_gradients(zip(gradients, trainable_variables))
        
        if step % (500) == 0:
            loss_v.append(loss)
            print("step %i, loss: %f" %(step, loss))
            
        
        x_feed, targ_real = samp_load.get_sup_data(bs = 6)
        # x_feed = tf.expand_dims(tf.stop_gradient(neural_net(x)), axis= 1)
        # x_feed = tf.stop_gradient(neural_net(x))
        targ_real = tf.expand_dims(targ_real, axis = 2)
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # gened_ppg = ppg_gen(x_feed, training= True)
            gened_ppg = ppg_gen(x_feed, training= True)
            gened_ppg = tf.expand_dims( gened_ppg, axis = 2)
            real_output = ppg_disc(targ_real, training=True)
            fake_output = ppg_disc(gened_ppg, training=True)
            gen_loss = generator_loss(fake_output)
            if step%5 ==0:
                disc_loss = discriminator_loss(real_output, fake_output)
 
        gradients_of_generator = gen_tape.gradient(gen_loss, ppg_gen.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, ppg_gen.trainable_variables))
        
        if step%5 ==0:
            gradients_of_discriminator = disc_tape.gradient(disc_loss, ppg_disc.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, ppg_disc.trainable_variables))
        
        if step % (500) == 0:
            # loss_v.append(loss)
            print("step %i, loss: %f, loss: %f" %(step, gen_loss, disc_loss))
            
        # if step % (9999) == 0:
#            wt_save(net, save_mdl = '../../../Dataset/Merl_Tim/NNsave/SavedWM/Models/ssl_mods')            
            
#%% Training main

with tf.device('gpu:0'): #very important
    trainNetProb(neural_net)
    
# with tf.device('gpu:0'): #very important
#     train_gen_model()

    
# wt_save(neural_net)
    
#%% Load Weight

def wt_load(neural_net, save_mdl = '../../../Dataset/Merl_Tim/NNsave/SavedWM/Models/ssl_modedfsdl'):
    neural_net.load_weights(save_mdl)
    return neural_net
    
neural_net =  wt_load(neural_net, save_mdl = '../../../Dataset/Merl_Tim/NNsave/SavedWM/Models/ssl_mods')

#%% data Plot
def data_plot(data, rs = 500, rw = 5, save_Fig = 0):
    opt = []
    for pos in list(range(rs, rs +rw*40, 40)):
        img = np.transpose(data[pos:pos+40:1,:,:],[1,2,0])
        img = tf.image.resize(img, (100, 100))/255.0
        opt.append(neural_net(img))
        
    a = np.array(opt).ravel().shape
    fig = plt.figure(figsize=(19.20*2,10.80*2))
    ax1 = fig.add_subplot()
    plt.plot(list(np.arange(0, a[0]/64,1/64)), np.array(opt).ravel(), linewidth=5)
    plt.ylabel('PPG (Normalized Voltage)', fontweight="bold", fontsize=35)
    plt.xlabel('Time (second)', fontweight="bold", fontsize=35)
    # plt.title('PPG plot',fontsize=45, fontweight="bold")
    plt.yticks(fontsize=35)
    plt.xticks(fontsize=35)
    plt.grid(visible=True, color='k', linestyle='-', linewidth=0.5)

    if save_Fig:
        plt.savefig('video_res.png', bbox_inches="tight", format = 'png', dpi= 500)
    plt.show()
    return np.array(opt).ravel()

def good_plot(x, y, x_label = 'Time (second)', y_label= 'PPG (Normalized Voltage)',  
              title = None, save_Fig = 0):
    
    fig = plt.figure(figsize=(19.20*2,10.80*2))
    ax1 = fig.add_subplot()
    plt.plot(x, y, linewidth=5)
    plt.ylabel(x_label, fontweight="bold", fontsize=35)
    plt.xlabel(y_label, fontweight="bold", fontsize=35)
    # plt.title('PPG plot',fontsize=45, fontweight="bold")
    plt.yticks(fontsize=35)
    plt.xticks(fontsize=35)
    plt.grid(visible=True, color='k', linestyle='-', linewidth=0.5)

    if save_Fig:
        plt.savefig('video_res.png', bbox_inches="tight", format = 'png', dpi= 500)
    plt.show()
    
    

#%% Frequency Domain analysis

from scipy.fft import fft, fftfreq

def freq_analysis(x):
    if len(x.shape)>1:
        print("Check the array shape. Should be one dimensional")
    yf = fft(x)
    xf = fftfreq(len(x), 1 / 64)

    # plt.plot(xf[0:int(len(x)/8)]*60, np.abs(yf)[0:int(len(x)/8)])
    good_plot(x = xf[0:int(len(x)/8)]*60, y =  np.abs(yf)[0:int(len(x)/8)])
    plt.show() 
    return 