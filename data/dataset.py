import os
import torch
from torch.utils.data import Dataset
from data.data_augmentation import *
import pandas as pd
import ast
import numpy as np
import nibabel as nib
import glob
join = os.path.join
import pandas as pd
import random

class Brats_loadall_nii(Dataset):
    def __init__(self, transforms='', root=None, num_cls=4, train_file='train.txt'):
        data_file_path = os.path.join('/work/H2020DeciderFicarra/asaporita/Progetto_AI_Medical_Imaging', train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()] #875 elements

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls

    def __getitem__(self, index):
        volpath = self.volpaths[index]
        name = self.names[index]
        
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...]

        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3)) # [B, C, H, W, D]

        _, H, W, Z = np.shape(y)
        y_flatten = np.reshape(y, (-1)) # Flatten the segmentation mask
        one_hot_targets = np.eye(self.num_cls)[y_flatten] # Convert to one-hot encoding where each voxel is represented as a vector of length num_cls
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1)) # Reshape back to 3D
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))  # Reorder dimensions
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        return x, y, yo, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_test_nii(Dataset):
    def __init__(self, transforms='', root=None, test_file='test.txt', num_cls=4):
        data_file_path = os.path.join('/work/H2020DeciderFicarra/asaporita/Progetto_AI_Medical_Imaging', test_file)

        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()] # 251 elements

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]

        x,y = self.transforms([x, y])

        # target required for models that require the segmentation as one-hot encoded targets, such as Dice loss.
        _, H, W, Z = np.shape(y)

        y_flatten = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y_flatten]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))
        yo = torch.squeeze(torch.from_numpy(yo), dim=0) 

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3)) # [B, C, H, W, D]
        y = np.ascontiguousarray(y)

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        return x, y, yo, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_val_nii(Dataset):
    def __init__(self, transforms='', root=None, val_file='val.txt', num_cls=4):
        data_file_path = os.path.join('/work/H2020DeciderFicarra/asaporita/Progetto_AI_Medical_Imaging', val_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths # 125 elements
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls


    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        # target required for models that require the segmentation as one-hot encoded targets, such as Dice loss.
        _, H, W, Z = np.shape(y)
        y_flatten = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y_flatten]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))
        yo = torch.squeeze(torch.from_numpy(yo), dim=0) 

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3)) # [B, C, H, W, D]
        y = np.ascontiguousarray(y)

        x = torch.squeeze(torch.from_numpy(x), dim=0) 
        y = torch.squeeze(torch.from_numpy(y), dim=0) 

        return x, y, yo, name

    def __len__(self):
        return len(self.volpaths)
