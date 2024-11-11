#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os


class SEQUENCE_DATASET(Dataset):
    def __init__(self,path_to_file,data,train,temporal_window):
        self.temporal_window = temporal_window
        self.X = []
        self.all_sample = []
        idx = 0
        for name in data :     
            temp = np.load(path_to_file+'/data/train/'+name)
            if idx == 0 :
                n_frames = temp.shape[1]
                mean = np.mean(temp, axis=None)
                std = np.std(temp, axis=None)
                temp = (temp.T - mean)/std
                temp = temp.T

            if idx == 1 :
                temp = temp[:,:n_frames]
            self.X.append(temp)
            idx +=1
    
        self.X = np.concatenate(self.X, axis=0)
        self.frames = np.asarray(list(range(n_frames)))
        
        if self.X.shape[0] > self.X.shape[1]:
            self.X=self.X.T
            
        self.data_points = len(self.X[0,:])
        brkpnts = []
        if train:
            train_brkpnts = np.load(os.path.join(path_to_file, 'results', 'train_breakpoints.npy'))
            for train_brkpnt in train_brkpnts :
                temp = list(range(train_brkpnt-self.temporal_window, train_brkpnt+1))
                brkpnts.extend(temp)
        else :
            test_brkpnts = np.load(os.path.join(path_to_file, 'results', 'test_breakpoints.npy'))
            for test_brkpnt in test_brkpnts :
                temp = list(range(test_brkpnt-self.temporal_window, test_brkpnt+1))
                brkpnts.extend(temp)

        self.prev = []
        self.count = 0
        self.seq_mean = np.mean(self.X[1:,:], axis=None)
        self.seq_std = np.std(self.X[1:,:], axis=None)
        self.vel_mean = np.mean(self.X[1,:], axis=None)
        self.vel_std = np.std(self.X[1,:], axis=None)
        temp = np.arange(self.data_points-self.temporal_window)
        self.choices = np.delete(temp, brkpnts, axis=0)
        np.random.shuffle(self.choices)
        self.idx = 0
        self.data_points = self.choices.shape[0]
        print('Initialize data. Datapoints %d' %self.data_points)

    def __len__(self):      
        return self.data_points

    def __getitem__(self, index):
        start = self.choices[self.idx]
        #start = np.random.choice(self.choices, replace=False)
        end = start+self.temporal_window
        velocity = self.X[1,start:end]
        sequence = self.X[1:,start:end]  
        frames = self.frames[start:end]
        self.idx += 1
        sequence = (sequence - self.seq_mean)/self.seq_std
        velocity = (velocity - self.vel_mean)/self.vel_std
        sample = {
                  'sequence' : torch.from_numpy(sequence),
                  'velocity' : torch.from_numpy(np.expand_dims(velocity, axis=0)),
                  'frames' : torch.from_numpy(np.expand_dims(frames, axis=0))
                 }
        #sequence = (sequence-self.mean)/self.std
            
        return sample
    
    
    
    
    
