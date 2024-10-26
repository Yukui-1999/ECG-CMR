import torch.utils.data as data
import pandas as pd
import os
import torch
import numpy as np
from typing import Any, Tuple
import matplotlib.pyplot as plt
from utils import augmentations, transformations
from torchvision import transforms
from utils.preprocess import process_snp
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler

class mutimodal_dataset(data.Dataset):
    def __init__(self,data_path=None,
                transform=True,
                augment=False,
                args=None,
                scaler=None,
                downstream=None,# downstream task, include classification and regression
                 ):
        super().__init__()
        self.downstream = downstream
        self.scaler = scaler
        self.transform=transform
        self.augment=augment
        self.args=args
        self.data = torch.load(data_path,map_location='cpu')
        prefix = data_path.split('/')[-1].split('_')[0]
        if prefix == 'trainval':
            prefix = 'train'
        self.ecg = self.data[str(prefix+'_ecg_data')]
        self.cmr = self.data[str(prefix+'_cmr_data')]
        self.tar = self.data[str(prefix+'_tar_data')]
        self.snp = process_snp(self.data[str(prefix+'_snp_data')])
        self.select_tar = self.data[str(prefix+'_select_tar_data')]
        # self.snp = torch.randint(0,2,(self.snp.shape[0],self.snp.shape[1],self.snp.shape[2]))
        #
        self.cha = self.data[str(prefix+'_cha_data')]
        self.I21 = self.data[str(prefix+'_I21_data')]
        self.I42 = self.data[str(prefix+'_I42_data')]
        self.I48 = self.data[str(prefix+'_I48_data')]
        self.I50 = self.data[str(prefix+'_I50_data')]
        if downstream == 'classification':
            if args.classification_dis == 'I21':
                self.label = np.array(self.I21.squeeze())
            elif args.classification_dis == 'I42':
                self.label = np.array(self.I42.squeeze())
            elif args.classification_dis == 'I48':
                self.label = np.array(self.I48.squeeze())
            elif args.classification_dis == 'I50':
                self.label = np.array(self.I50.squeeze())
        if augment:
            self.scaler = StandardScaler()
            self.cha = self.scaler.fit_transform(self.cha)
        else:
            self.cha = self.scaler.transform(self.cha)
        

        print(f'ecg.shape: {self.ecg.shape}, cmr.shape: {self.cmr.shape}, tar.shape: {self.tar.shape}, snp.shape: {self.snp.shape}, cha.shape: {self.cha.shape}, I21.shape: {self.I21.shape}, I42.shape: {self.I42.shape}, I48.shape: {self.I48.shape}, I50.shape: {self.I50.shape}')
        # print(self.snp[0][0])
        # print(self.snp[0][1])
    def get_labels(self):   
        return self.label 
    def get_scaler(self):
        return self.scaler
    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.ecg)

    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
        ecg = self.ecg[index]
        cmr = self.cmr[index]
        tar = self.tar[index]
        snp = self.snp[index]
        cha = self.cha[index]
        I21 = self.I21[index]
        I42 = self.I42[index]
        I48 = self.I48[index]
        I50 = self.I50[index]
        select_tar = self.select_tar[index]
        if self.transform == True and self.augment == False:# val and test stage
            
            ecg_transform = transforms.Compose([
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_transform(ecg)
            
            cmr_transform = transforms.Compose([
                transforms.Resize(size=self.args.resizeshape, antialias=True),
                transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
            ])
            cmr = cmr_transform(cmr)
        if self.augment == True:# train stage
            ecg_augment = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.ecg_input_size[-1], resize=False),
                augmentations.TimeFlip(prob=self.args.timeFlip),
                augmentations.SignFlip(prob=self.args.signFlip),
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_augment(ecg)
            
            cmr_augment = transforms.Compose([
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(size=self.args.resizeshape, scale=(0.8, 1.0), ratio=(0.9, 1.1),antialias=True),
                transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
            ])
            cmr = cmr_augment(cmr)
        
        return ecg,cmr,tar,snp,cha,I21,I42,I48,I50,select_tar
    

# class mutimodal_dataset(data.Dataset):
#     def __init__(self,train_path=None,
#                 val_path=None,
#                 test_path=None,
#                 type='train',
#                 transform=True,
#                 augment=False,
#                 args=None,
#                  ):
#         super().__init__()
#         self.type=type
#         self.transform=transform
#         self.augment=augment
#         self.args=args
#         # print(f'args: {args.input_size}')
#         self.train_csv = pd.read_csv(train_path,dtype={176: str})
#         self.val_csv = pd.read_csv(val_path,dtype={176: str})
#         self.test_csv = pd.read_csv(test_path,dtype={176: str})

#         self.train_ecg = self.train_csv['20205_2_0']
#         self.val_ecg = self.val_csv['20205_2_0']
#         self.test_ecg = self.test_csv['20205_2_0']

#         self.train_cmr = self.train_csv['20209_2_0']
#         self.val_cmr = self.val_csv['20209_2_0']
#         self.test_cmr = self.test_csv['20209_2_0']


#         self.train_tar , self.val_tar , self.test_tar = get_tar(self.train_csv,self.val_csv,self.test_csv)
#         # print(self.train_tar.info())
#         # print(self.val_tar.info())
#         # print(self.test_tar.info())
    
#     def __len__(self) -> int:
#         """return the number of samples in the dataset"""
#         if self.type == 'train':
#             return len(self.train_csv)
#         elif self.type == 'val':
#             return len(self.val_csv)
#         elif self.type == 'test':
#             return len(self.test_csv)

#     def __getitem__(self, index) -> Any:
#         """return the sample with given index"""
#         if self.type == 'train':
#             ecg_path = self.train_ecg[index]
#             cmr_path = self.train_cmr[index]
#             tar = self.train_tar.iloc[index]
#         if self.type == 'val':
#             ecg_path = self.val_ecg[index]
#             cmr_path = self.val_cmr[index]
#             tar = self.val_tar.iloc[index]
#         if self.type == 'test':
#             ecg_path = self.test_ecg[index]
#             cmr_path = self.test_cmr[index]
#             tar = self.test_tar.iloc[index]
        
#         ecg_path = os.path.join('/mnt/data/ukb_heartmri/ukb_20205', ecg_path)
#         ecg = get_ecg(ecg_path)
#         ecg = torch.from_numpy(ecg).float()
#         if self.transform == True:
#             transform = transforms.Compose([
#                 augmentations.CropResizing(lower_bnd=1,upper_bnd=1,fixed_crop_len=None, start_idx=0, resize=True),
#                 transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise")
#             ])
#             ecg = transform(ecg)

#         if self.augment == True:
#             augment = transforms.Compose([
#                 augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False),
#                 augmentations.TimeFlip(prob=self.args.timeFlip),
#                 augmentations.SignFlip(prob=self.args.signFlip),
#                 transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
#             ])
#             ecg = augment(ecg)
        
        
#         img = get_img(cmr_path)
#         img = torch.from_numpy(img).float()
#         tar = torch.from_numpy(np.array(tar)).float()
#         return ecg,img,tar
        
        
# import argparse

# # 创建 ArgumentParser 对象
# parser = argparse.ArgumentParser(description='Your description here')

# # 添加参数
# parser.add_argument('--input_size', type=tuple, default=(12, 5000))
# parser.add_argument('--timeFlip', type=float, default=0.33)
# parser.add_argument('--signFlip', type=float, default=0.33)

# # 解析参数
# args = parser.parse_args()

# # 现在你可以使用 args.input_size, args.ft_surr_phase_noise 等来访问参数值
# dataset = mutimodal_dataset(data_path='/mnt/data/dingzhengyao/work/checkpoint/preject_version1/data/val_data_dict_v5.pt',
#                             transform=True,
#                             augment=True,
#                             args=args
#                             )
# ecg,cmr,tar = dataset.__getitem__(0)
# print(ecg.shape)
# print(f'ecg.min: {ecg.min()}, ecg.max: {ecg.max()}')
# print(cmr.shape)
# print(f'cmr.min: {cmr.min()}, cmr.max: {cmr.max()}')
# print(tar.shape)