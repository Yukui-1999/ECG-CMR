import torch.utils.data as data
import pandas as pd
import os
import torch
import numpy as np
from typing import Any, Tuple
import matplotlib.pyplot as plt
from src.utils import augmentations, transformations
from torchvision import transforms
from src.utils.preprocess import get_ecg,get_img,get_tar


class mutimodal_dataset(data.Dataset):
    def __init__(self,train_path=None,
                val_path=None,
                test_path=None,
                type='train',
                transform=True,
                augment=False,
                args=None,
                 ):
        super().__init__()
        self.type=type
        self.transform=transform
        self.augment=augment
        self.args=args
        print(f'args: {args.input_size}')
        self.train_csv = pd.read_csv(train_path)
        self.val_csv = pd.read_csv(val_path)
        self.test_csv = pd.read_csv(test_path)

        self.train_ecg = self.train_csv['20205_2_0']
        self.val_ecg = self.val_csv['20205_2_0']
        self.test_ecg = self.test_csv['20205_2_0']

        self.train_cmr = self.train_csv['20209_2_0']
        self.val_cmr = self.val_csv['20209_2_0']
        self.test_cmr = self.test_csv['20209_2_0']


        self.train_tar , self.val_tar , self.test_tar = get_tar(self.train_csv,self.val_csv,self.test_csv)
        # print(self.train_tar.info())
        # print(self.val_tar.info())
        # print(self.test_tar.info())
    
    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        if self.type == 'train':
            return len(self.train_csv)
        elif self.type == 'val':
            return len(self.val_csv)
        elif self.type == 'test':
            return len(self.test_csv)

    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
        if self.type == 'train':
            ecg_path = self.train_ecg[index]
            cmr_path = self.train_cmr[index]
            tar = self.train_tar.iloc[index]
        if self.type == 'val':
            ecg_path = self.val_ecg[index]
            cmr_path = self.val_cmr[index]
            tar = self.val_tar.iloc[index]
        if self.type == 'test':
            ecg_path = self.test_ecg[index]
            cmr_path = self.test_cmr[index]
            tar = self.test_tar.iloc[index]
        
        ecg_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20205', ecg_path)
        ecg = get_ecg(ecg_path)
        ecg = torch.from_numpy(ecg).float()
        if self.transform == True:
            transform = transforms.Compose([
                augmentations.CropResizing(lower_bnd=1,upper_bnd=1,fixed_crop_len=None, start_idx=0, resize=True),
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise")
            ])
            ecg = transform(ecg)

        if self.augment == True:
            augment = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False),
                augmentations.TimeFlip(prob=self.args.timeFlip),
                augmentations.SignFlip(prob=self.args.signFlip),
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = augment(ecg)
        
        
        img = get_img(cmr_path)
        img = torch.from_numpy(img).float()
        tar = torch.from_numpy(np.array(tar)).float()
        return ecg,img,tar
        
        
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Your description here')

# 添加参数
parser.add_argument('--input_size', type=tuple, default=(12, 5000))
parser.add_argument('--timeFlip', type=float, default=0.33)
parser.add_argument('--signFlip', type=float, default=0.33)

# 解析参数
args = parser.parse_args()

# 现在你可以使用 args.input_size, args.ft_surr_phase_noise 等来访问参数值
dataset = mutimodal_dataset(train_path='/mnt/data/ukb_collation/ukb_ecg_cmr/data/train_v2.csv',
                            val_path='/mnt/data/ukb_collation/ukb_ecg_cmr/data/val_v2.csv',
                            test_path='/mnt/data/ukb_collation/ukb_ecg_cmr/data/test_v2.csv',
                            type='train',
                            transform=True,
                            augment=True,
                            args=args
                            )
ecg,cmr,tar = dataset.__getitem__(0)
print(ecg.shape)
print(f'ecg.min: {ecg.min()}, ecg.max: {ecg.max()}')
print(cmr.shape)
print(f'cmr.min: {cmr.min()}, cmr.max: {cmr.max()}')
print(tar.shape)