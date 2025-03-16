import os
from typing import Tuple
import sys

import numpy as np
import scipy.io
import torch
import torch.utils.data as data
import glob
import random
import math
import time
import scipy.io
import numpy as np
import cv2

from classes.fc4.repvit.utils1 import hwc_chw, gamma_correct, brg_to_rgb
from auxiliary.utils import normalize, bgr_to_rgb, linear_to_nonlinear, hwc_to_chw
from classes.data.DataAugmenter import DataAugmenterseq



class CtaDataset():

    def __init__(self, mode: str = "train", input_size: Tuple = (224, 224), device: int=1):
        dataset_device = ['HuaweiMate30', 'HuaweiP30PRO', 'iphone14pm', 'vivoiqooneo5', 'Xiaomi11PRO', 'Xiaomi13']
        num_device = ['mate30', 'P30pro', 'iphonepm', 'vivo', 'xiaomi11pro', 'xiaomi13']
        self.mode = mode
        self.__input_size = input_size
        self.__da = DataAugmenterseq(self.__input_size)
        self._paths_to_seqs = []
        self._paths_to_seqs = []
        self._nums_to_seqs = []

        if self.mode=="train":
            path_to_dataset = '/mnt/disk1/NPY2/' + dataset_device[device-1] +'/'
            train_path = './dataset/CTA/train_'+num_device[device-1]+'.npy'
            train_info = np.load(train_path, allow_pickle=True).item()
            train_ids = train_info['id']
            train_nums = train_info['num']
            for i in range(len(train_ids)):
                id = train_ids[i]
                num = train_nums[i]
                for j in range(1, num+1):
                    self._paths_to_seqs.append(path_to_dataset + str(id) + ',' + str(j))
                    self._nums_to_seqs.append(num)
        else:
            path_to_dataset = '/mnt/disk1/NPY2/' + dataset_device[device-1] +'/'
            train_path = './dataset/CTA/test_'+num_device[device-1]+'.npy'
            train_info = np.load(train_path, allow_pickle=True).item()
            train_ids = train_info['id']
            train_nums = train_info['num']
            for i in range(len(train_ids)):
                id = train_ids[i]
                num = train_nums[i]
                for j in range(1, num+1):
                    self._paths_to_seqs.append(path_to_dataset + str(id) + ',' + str(j))
                    self._nums_to_seqs.append(num)
    
    def __getitem__(self, index: int) -> Tuple:
        path_to_sequence = self._paths_to_seqs[index]
        num_to_sequence = self._nums_to_seqs[index]
        path_to_frame = str(path_to_sequence.split(',')[0])
        label_path = path_to_frame + '/illu_mat.npy'
        illums = np.load(label_path, allow_pickle=True).item()
        id = int(path_to_sequence.split(',')[-1])
        files_seq = []
        if id == 1:
            files_seq.append(path_to_frame+'/'+str(id)+'.dng.npy')
            files_seq.append(path_to_frame+'/'+str(id)+'.dng.npy')
            files_seq.append(path_to_frame+'/'+str(id+1)+'.dng.npy')     
        elif id == num_to_sequence:
            files_seq.append(path_to_frame+'/'+str(id-1)+'.dng.npy')
            files_seq.append(path_to_frame+'/'+str(id)+'.dng.npy')
            files_seq.append(path_to_frame+'/'+str(id)+'.dng.npy')  
        else:
            files_seq.append(path_to_frame+'/'+str(id-1)+'.dng.npy')
            files_seq.append(path_to_frame+'/'+str(id)+'.dng.npy')
            files_seq.append(path_to_frame+'/'+str(id+1)+'.dng.npy')  
        images = [np.array(np.load(file), dtype='float32') for file in files_seq]
        seq = np.array(images, dtype='float32')
        illuminant = np.array(illums[str(id)], dtype='float32')

        mimic = torch.from_numpy(self.__da.augment_mimic(seq).transpose((0, 3, 1, 2)).copy())
        

        if self.mode == "train":
            seq, color_bias = self.__da.augment_sequence(seq, illuminant)
            color_bias = np.array([[[color_bias[0][0], color_bias[1][1], color_bias[2][2]]]], dtype=np.float32)
            mimic = torch.mul(mimic, torch.from_numpy(color_bias).view(1, 3, 1, 1))
            mimic = np.clip(mimic, 0.0, 255.0) * (1.0 / 255)
        else:
            seq = self.__da.resize_sequence(seq)

        #seq = self.__da.resize_sequence(seq)

        seq = np.clip(seq, 0.0, 255.0) * (1.0 / 255)
        seq = hwc_chw(gamma_correct(brg_to_rgb(seq)))

        seq = torch.from_numpy(seq.copy())
        illuminant = torch.from_numpy(illuminant.copy())

        return seq, mimic, illuminant, path_to_sequence
        #return seq, path_to_sequence

    def __len__(self) -> int:
        return len(self._paths_to_seqs)
    
