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
import torch.nn.functional as F
import pickle
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
        self.cha = self.data[str(prefix+'_cha_data')]
        self.I21 = self.data[str(prefix+'_I21_data')]
        self.I42 = self.data[str(prefix+'_I42_data')]
        self.I48 = self.data[str(prefix+'_I48_data')]
        self.I50 = self.data[str(prefix+'_I50_data')]
        self.I08 = self.data[str(prefix+'_I08_data')]
        self.I25 = self.data[str(prefix+'_I25_data')]
        self.I34 = self.data[str(prefix+'_I34_data')]
        self.I35 = self.data[str(prefix+'_I35_data')]
        self.check_time = self.data[str(prefix+'_check_time')]
        self.la_cmr = self.data[str(prefix+'_la_cmr_data')]

        if downstream == 'classification':
            if args.classification_dis == 'I21':
                self.label = np.array([1 if x != '0' else 0 for x in self.I21])
            elif args.classification_dis == 'I42':
                self.label = np.array([1 if x != '0' else 0 for x in self.I42])
            elif args.classification_dis == 'I48':
                self.label = np.array([1 if x != '0' else 0 for x in self.I48])
            elif args.classification_dis == 'I50':
                self.label = np.array([1 if x != '0' else 0 for x in self.I50])
            elif args.classification_dis == 'I08':
                self.label = np.array([1 if x != '0' else 0 for x in self.I08])
            elif args.classification_dis == 'I25':
                self.label = np.array([1 if x != '0' else 0 for x in self.I25])
            elif args.classification_dis == 'I34':
                self.label = np.array([1 if x != '0' else 0 for x in self.I34])
            elif args.classification_dis == 'I35':
                self.label = np.array([1 if x != '0' else 0 for x in self.I35])
            #统计label中1和0的个数
            count_ones = np.sum(self.label == 1)
            count_zeros = np.sum(self.label == 0)
            print(f"Number of 1s: {count_ones}")
            print(f"Number of 0s: {count_zeros}")
        if augment:
            self.scaler = StandardScaler()
            self.cha = self.scaler.fit_transform(self.cha)
        else:
            self.cha = self.scaler.transform(self.cha)
        

        print(f'''ecg.shape: {self.ecg.shape}, cmr.shape: {self.cmr.shape},la_cmr.shape: {len(self.la_cmr)}, 
              tar.shape: {self.tar.shape}, snp.shape: {self.snp.shape}, cha.shape: {self.cha.shape}, 
              I21.shape: {len(self.I21)}, I42.shape: {len(self.I42)}, I48.shape: {len(self.I48)}, 
              I50.shape: {len(self.I50)}, I08.shape: {len(self.I08)}, I25.shape: {len(self.I25)},
              I34.shape: {len(self.I34)}, I35.shape: {len(self.I35)}, check_time.shape: {len(self.check_time)}''')
        # print(self.snp[0][0])
        # print(self.snp[0][1])
    def get_labels(self):   
        return self.label 
    def get_scaler(self):
        return self.scaler
    def add_random_noise(self, image, noise_range=(-0.1, 0.1)):
        noise = torch.FloatTensor(image.size()).uniform_(*noise_range)
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, -1, 1)  # 确保像素值在 [-1, 1] 范围内
        return noisy_image
    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.ecg)

    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
        ecg = self.ecg[index]
        cmr = self.cmr[index]
        la_cmr = torch.from_numpy(self.la_cmr[index]).float()
        tar = self.tar[index]
        snp = self.snp[index]
        cha = self.cha[index]
        I21 = self.I21[index]
        I42 = self.I42[index]
        I48 = self.I48[index]
        I50 = self.I50[index]
        I08 = self.I08[index]
        I25 = self.I25[index]
        I34 = self.I34[index]
        I35 = self.I35[index]
        check_time = self.check_time[index]
        select_tar = self.select_tar[index]

        if self.transform == True and self.augment == False:# val and test stage
            
            ecg_transform = transforms.Compose([
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_transform(ecg)

            la_cmr = (la_cmr-la_cmr.min())/(la_cmr.max()-la_cmr.min())
            if self.args.resizeshape is not None:
                cmr_transform = transforms.Compose([
                    transforms.Resize(size=self.args.resizeshape),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_transform = transforms.Compose([
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            cmr = cmr_transform(cmr)
            la_cmr = cmr_transform(la_cmr)
            
            

        if self.augment == True:# train stage
            ecg_augment = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.ecg_input_size[-1], resize=False),
                augmentations.TimeFlip(prob=self.args.timeFlip),
                augmentations.SignFlip(prob=self.args.signFlip),
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_augment(ecg)

            assert cmr.shape[0] == 50 and cmr.shape[1] == 80 and cmr.shape[2] == 80
            assert la_cmr.shape[0] == 50 and la_cmr.shape[1] == 96 and la_cmr.shape[2] == 96
            la_cmr = (la_cmr-la_cmr.min())/(la_cmr.max()-la_cmr.min())
            if self.args.resizeshape is not None:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(size=self.args.resizeshape, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            cmr = cmr_augment(cmr)
            la_cmr = cmr_augment(la_cmr)
            
            
        
        batch = {
            'ecg': ecg,
            'cmr': cmr,
            'tar': tar,
            'snp': snp,
            'cha': cha,
            'I21': I21,
            'I42': I42,
            'I48': I48,
            'I50': I50,
            'select_tar': select_tar,
            'la_cmr': la_cmr,
            'I08': I08,
            'I25': I25,
            'I34': I34,
            'I35': I35,
            'check_time': check_time
        }

        return batch
    



class mutimodal_dataset_ECGCMR(data.Dataset):
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
        self.cha = self.data[str(prefix+'_cha_data')]
        self.I21 = self.data[str(prefix+'_I21_data')]
        self.I42 = self.data[str(prefix+'_I42_data')]
        self.I48 = self.data[str(prefix+'_I48_data')]
        self.I50 = self.data[str(prefix+'_I50_data')]
        self.I08 = self.data[str(prefix+'_I08_data')]
        self.I25 = self.data[str(prefix+'_I25_data')]
        self.I34 = self.data[str(prefix+'_I34_data')]
        self.I35 = self.data[str(prefix+'_I35_data')]
        self.check_time = self.data[str(prefix+'_check_time')]
        

        if downstream == 'classification':
            if args.classification_dis == 'I21':
                self.label = np.array([1 if x != '0' else 0 for x in self.I21])
            elif args.classification_dis == 'I42':
                self.label = np.array([1 if x != '0' else 0 for x in self.I42])
            elif args.classification_dis == 'I48':
                self.label = np.array([1 if x != '0' else 0 for x in self.I48])
            elif args.classification_dis == 'I50':
                self.label = np.array([1 if x != '0' else 0 for x in self.I50])
            elif args.classification_dis == 'I08':
                self.label = np.array([1 if x != '0' else 0 for x in self.I08])
            elif args.classification_dis == 'I25':
                self.label = np.array([1 if x != '0' else 0 for x in self.I25])
            elif args.classification_dis == 'I34':
                self.label = np.array([1 if x != '0' else 0 for x in self.I34])
            elif args.classification_dis == 'I35':
                self.label = np.array([1 if x != '0' else 0 for x in self.I35])
            #统计label中1和0的个数
            count_ones = np.sum(self.label == 1)
            count_zeros = np.sum(self.label == 0)
            print(f"Number of 1s: {count_ones}")
            print(f"Number of 0s: {count_zeros}")
        if augment:
            self.scaler = StandardScaler()
            self.cha = self.scaler.fit_transform(self.cha)
        else:
            self.cha = self.scaler.transform(self.cha)
        

        print(f'''ecg.shape: {self.ecg.shape}, cmr.shape: {self.cmr.shape},
              tar.shape: {self.tar.shape}, snp.shape: {self.snp.shape}, cha.shape: {self.cha.shape}, 
              I21.shape: {len(self.I21)}, I42.shape: {len(self.I42)}, I48.shape: {len(self.I48)}, 
              I50.shape: {len(self.I50)}, I08.shape: {len(self.I08)}, I25.shape: {len(self.I25)},
              I34.shape: {len(self.I34)}, I35.shape: {len(self.I35)}, check_time.shape: {len(self.check_time)}''')
        # print(self.snp[0][0])
        # print(self.snp[0][1])
    def get_labels(self):   
        return self.label 
    def get_scaler(self):
        return self.scaler
    def add_random_noise(self, image, noise_range=(-0.1, 0.1)):
        noise = torch.FloatTensor(image.size()).uniform_(*noise_range)
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, -1, 1)  # 确保像素值在 [-1, 1] 范围内
        return noisy_image
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
        I08 = self.I08[index]
        I25 = self.I25[index]
        I34 = self.I34[index]
        I35 = self.I35[index]
        check_time = self.check_time[index]
        select_tar = self.select_tar[index]

        if self.transform == True and self.augment == False:# val and test stage
            
            ecg_transform = transforms.Compose([
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_transform(ecg)

            
            if self.args.resizeshape is not None:
                cmr_transform = transforms.Compose([
                    transforms.Resize(size=self.args.resizeshape),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_transform = transforms.Compose([
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

            assert cmr.shape[0] == 50 and cmr.shape[1] == 80 and cmr.shape[2] == 80
            
            if self.args.resizeshape is not None:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(size=self.args.resizeshape, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            cmr = cmr_augment(cmr)
            
            
        batch = {
            'ecg': ecg,
            'cmr': cmr,
            'tar': tar,
            'snp': snp,
            'cha': cha,
            'I21': I21,
            'I42': I42,
            'I48': I48,
            'I50': I50,
            'select_tar': select_tar,
            'I08': I08,
            'I25': I25,
            'I34': I34,
            'I35': I35,
            'check_time': check_time
        }
         
        return batch
    


class mutimodal_dataset_ECGlaCMR(data.Dataset):
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
        
        self.tar = self.data[str(prefix+'_tar_data')]
        self.snp = process_snp(self.data[str(prefix+'_snp_data')])
        self.select_tar = self.data[str(prefix+'_select_tar_data')]
        self.cha = self.data[str(prefix+'_cha_data')]
        self.I21 = self.data[str(prefix+'_I21_data')]
        self.I42 = self.data[str(prefix+'_I42_data')]
        self.I48 = self.data[str(prefix+'_I48_data')]
        self.I50 = self.data[str(prefix+'_I50_data')]
        self.I08 = self.data[str(prefix+'_I08_data')]
        self.I25 = self.data[str(prefix+'_I25_data')]
        self.I34 = self.data[str(prefix+'_I34_data')]
        self.I35 = self.data[str(prefix+'_I35_data')]
        self.check_time = self.data[str(prefix+'_check_time')]
        self.la_cmr = self.data[str(prefix+'_la_cmr_data')]

        if downstream == 'classification':
            if args.classification_dis == 'I21':
                self.label = np.array([1 if x != '0' else 0 for x in self.I21])
            elif args.classification_dis == 'I42':
                self.label = np.array([1 if x != '0' else 0 for x in self.I42])
            elif args.classification_dis == 'I48':
                self.label = np.array([1 if x != '0' else 0 for x in self.I48])
            elif args.classification_dis == 'I50':
                self.label = np.array([1 if x != '0' else 0 for x in self.I50])
            elif args.classification_dis == 'I08':
                self.label = np.array([1 if x != '0' else 0 for x in self.I08])
            elif args.classification_dis == 'I25':
                self.label = np.array([1 if x != '0' else 0 for x in self.I25])
            elif args.classification_dis == 'I34':
                self.label = np.array([1 if x != '0' else 0 for x in self.I34])
            elif args.classification_dis == 'I35':
                self.label = np.array([1 if x != '0' else 0 for x in self.I35])
            #统计label中1和0的个数
            count_ones = np.sum(self.label == 1)
            count_zeros = np.sum(self.label == 0)
            print(f"Number of 1s: {count_ones}")
            print(f"Number of 0s: {count_zeros}")
        if augment:
            self.scaler = StandardScaler()
            self.cha = self.scaler.fit_transform(self.cha)
        else:
            self.cha = self.scaler.transform(self.cha)
        

        print(f'''ecg.shape: {self.ecg.shape}, la_cmr.shape: {len(self.la_cmr)}, 
              tar.shape: {self.tar.shape}, snp.shape: {self.snp.shape}, cha.shape: {self.cha.shape}, 
              I21.shape: {len(self.I21)}, I42.shape: {len(self.I42)}, I48.shape: {len(self.I48)}, 
              I50.shape: {len(self.I50)}, I08.shape: {len(self.I08)}, I25.shape: {len(self.I25)},
              I34.shape: {len(self.I34)}, I35.shape: {len(self.I35)}, check_time.shape: {len(self.check_time)}''')
        # print(self.snp[0][0])
        # print(self.snp[0][1])
    def get_labels(self):   
        return self.label 
    def get_scaler(self):
        return self.scaler
    def add_random_noise(self, image, noise_range=(-0.1, 0.1)):
        noise = torch.FloatTensor(image.size()).uniform_(*noise_range)
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, -1, 1)  # 确保像素值在 [-1, 1] 范围内
        return noisy_image
    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.ecg)

    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
        ecg = self.ecg[index]
        
        la_cmr = torch.from_numpy(self.la_cmr[index]).float()
        tar = self.tar[index]
        snp = self.snp[index]
        cha = self.cha[index]
        I21 = self.I21[index]
        I42 = self.I42[index]
        I48 = self.I48[index]
        I50 = self.I50[index]
        I08 = self.I08[index]
        I25 = self.I25[index]
        I34 = self.I34[index]
        I35 = self.I35[index]
        check_time = self.check_time[index]
        select_tar = self.select_tar[index]

        if self.transform == True and self.augment == False:# val and test stage
            
            ecg_transform = transforms.Compose([
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_transform(ecg)

            la_cmr = (la_cmr-la_cmr.min())/(la_cmr.max()-la_cmr.min())
            if self.args.resizeshape is not None:
                cmr_transform = transforms.Compose([
                    transforms.Resize(size=self.args.resizeshape),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_transform = transforms.Compose([
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
           
            la_cmr = cmr_transform(la_cmr)
            

        if self.augment == True:# train stage
            ecg_augment = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.ecg_input_size[-1], resize=False),
                augmentations.TimeFlip(prob=self.args.timeFlip),
                augmentations.SignFlip(prob=self.args.signFlip),
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_augment(ecg)

           
            assert la_cmr.shape[0] == 50 and la_cmr.shape[1] == 96 and la_cmr.shape[2] == 96
            la_cmr = (la_cmr-la_cmr.min())/(la_cmr.max()-la_cmr.min())
            if self.args.resizeshape is not None:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(size=self.args.resizeshape, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
           
            la_cmr = cmr_augment(la_cmr)
            
          
            
        batch = {
                'ecg': ecg,
                'la_cmr': la_cmr,
                'tar': tar,
                'snp': snp,
                'cha': cha,
                'I21': I21,
                'I42': I42,
                'I48': I48,
                'I50': I50,
                'select_tar': select_tar,
                'I08': I08,
                'I25': I25,
                'I34': I34,
                'I35': I35,
                'check_time': check_time
            }
        return batch
    


class mutimodal_dataset_CMRlaCMR(data.Dataset):
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
        
        self.cmr = self.data[str(prefix+'_cmr_data')]
        self.tar = self.data[str(prefix+'_tar_data')]
        self.snp = process_snp(self.data[str(prefix+'_snp_data')])
        self.select_tar = self.data[str(prefix+'_select_tar_data')]
        self.cha = self.data[str(prefix+'_cha_data')]
        self.I21 = self.data[str(prefix+'_I21_data')]
        self.I42 = self.data[str(prefix+'_I42_data')]
        self.I48 = self.data[str(prefix+'_I48_data')]
        self.I50 = self.data[str(prefix+'_I50_data')]
        self.I08 = self.data[str(prefix+'_I08_data')]
        self.I25 = self.data[str(prefix+'_I25_data')]
        self.I34 = self.data[str(prefix+'_I34_data')]
        self.I35 = self.data[str(prefix+'_I35_data')]
        self.check_time = self.data[str(prefix+'_check_time')]
        self.la_cmr = self.data[str(prefix+'_la_cmr_data')]

        if downstream == 'classification':
            if args.classification_dis == 'I21':
                self.label = np.array([1 if x != '0' else 0 for x in self.I21])
            elif args.classification_dis == 'I42':
                self.label = np.array([1 if x != '0' else 0 for x in self.I42])
            elif args.classification_dis == 'I48':
                self.label = np.array([1 if x != '0' else 0 for x in self.I48])
            elif args.classification_dis == 'I50':
                self.label = np.array([1 if x != '0' else 0 for x in self.I50])
            elif args.classification_dis == 'I08':
                self.label = np.array([1 if x != '0' else 0 for x in self.I08])
            elif args.classification_dis == 'I25':
                self.label = np.array([1 if x != '0' else 0 for x in self.I25])
            elif args.classification_dis == 'I34':
                self.label = np.array([1 if x != '0' else 0 for x in self.I34])
            elif args.classification_dis == 'I35':
                self.label = np.array([1 if x != '0' else 0 for x in self.I35])
            #统计label中1和0的个数
            count_ones = np.sum(self.label == 1)
            count_zeros = np.sum(self.label == 0)
            print(f"Number of 1s: {count_ones}")
            print(f"Number of 0s: {count_zeros}")
        if augment:
            self.scaler = StandardScaler()
            self.cha = self.scaler.fit_transform(self.cha)
        else:
            self.cha = self.scaler.transform(self.cha)
        

        print(f''' cmr.shape: {self.cmr.shape},la_cmr.shape: {len(self.la_cmr)}, 
              tar.shape: {self.tar.shape}, snp.shape: {self.snp.shape}, cha.shape: {self.cha.shape}, 
              I21.shape: {len(self.I21)}, I42.shape: {len(self.I42)}, I48.shape: {len(self.I48)}, 
              I50.shape: {len(self.I50)}, I08.shape: {len(self.I08)}, I25.shape: {len(self.I25)},
              I34.shape: {len(self.I34)}, I35.shape: {len(self.I35)}, check_time.shape: {len(self.check_time)}''')
        # print(self.snp[0][0])
        # print(self.snp[0][1])
    def get_labels(self):   
        return self.label 
    def get_scaler(self):
        return self.scaler
    def add_random_noise(self, image, noise_range=(-0.1, 0.1)):
        noise = torch.FloatTensor(image.size()).uniform_(*noise_range)
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, -1, 1)  # 确保像素值在 [-1, 1] 范围内
        return noisy_image
    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.cmr)

    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
       
        cmr = self.cmr[index]
        la_cmr = torch.from_numpy(self.la_cmr[index]).float()
        tar = self.tar[index]
        snp = self.snp[index]
        cha = self.cha[index]
        I21 = self.I21[index]
        I42 = self.I42[index]
        I48 = self.I48[index]
        I50 = self.I50[index]
        I08 = self.I08[index]
        I25 = self.I25[index]
        I34 = self.I34[index]
        I35 = self.I35[index]
        check_time = self.check_time[index]
        select_tar = self.select_tar[index]

        if self.transform == True and self.augment == False:# val and test stage
            
            

            la_cmr = (la_cmr-la_cmr.min())/(la_cmr.max()-la_cmr.min())
            if self.args.resizeshape is not None:
                cmr_transform = transforms.Compose([
                    transforms.Resize(size=self.args.resizeshape),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_transform = transforms.Compose([
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            cmr = cmr_transform(cmr)
            la_cmr = cmr_transform(la_cmr)
            
            

        if self.augment == True:# train stage
            

            assert cmr.shape[0] == 50 and cmr.shape[1] == 80 and cmr.shape[2] == 80
            assert la_cmr.shape[0] == 50 and la_cmr.shape[1] == 96 and la_cmr.shape[2] == 96
            la_cmr = (la_cmr-la_cmr.min())/(la_cmr.max()-la_cmr.min())
            if self.args.resizeshape is not None:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(size=self.args.resizeshape, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            cmr = cmr_augment(cmr)
            la_cmr = cmr_augment(la_cmr)
            
            
            
        
        batch = {
                'cmr': cmr,
                'la_cmr': la_cmr,
                'tar': tar,
                'snp': snp,
                'cha': cha,
                'I21': I21,
                'I42': I42,
                'I48': I48,
                'I50': I50,
                'select_tar': select_tar,
                'I08': I08,
                'I25': I25,
                'I34': I34,
                'I35': I35,
                'check_time': check_time
            }
        return batch
    



class mutimodal_dataset_ECG(data.Dataset):
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
        self.eid = self.data[str(prefix+'_eid')]
        self.tar = self.data[str(prefix+'_tar_data')]
        self.snp = process_snp(self.data[str(prefix+'_snp_data')])
        self.select_tar = self.data[str(prefix+'_select_tar_data')]
        self.cha = self.data[str(prefix+'_cha_data')]
        self.I21 = self.data[str(prefix+'_I21_data')]
        self.I42 = self.data[str(prefix+'_I42_data')]
        self.I48 = self.data[str(prefix+'_I48_data')]
        self.I50 = self.data[str(prefix+'_I50_data')]
        self.I08 = self.data[str(prefix+'_I08_data')]
        self.I25 = self.data[str(prefix+'_I25_data')]
        self.I34 = self.data[str(prefix+'_I34_data')]
        self.I35 = self.data[str(prefix+'_I35_data')]
        self.check_time = self.data[str(prefix+'_check_time')]
        

        if downstream == 'classification':
            if args.classification_dis == 'I21':
                self.label = np.array([1 if x != '0' else 0 for x in self.I21])
            elif args.classification_dis == 'I42':
                self.label = np.array([1 if x != '0' else 0 for x in self.I42])
            elif args.classification_dis == 'I48':
                self.label = np.array([1 if x != '0' else 0 for x in self.I48])
            elif args.classification_dis == 'I50':
                self.label = np.array([1 if x != '0' else 0 for x in self.I50])
            elif args.classification_dis == 'I08':
                self.label = np.array([1 if x != '0' else 0 for x in self.I08])
            elif args.classification_dis == 'I25':
                self.label = np.array([1 if x != '0' else 0 for x in self.I25])
            elif args.classification_dis == 'I34':
                self.label = np.array([1 if x != '0' else 0 for x in self.I34])
            elif args.classification_dis == 'I35':
                self.label = np.array([1 if x != '0' else 0 for x in self.I35])
            #统计label中1和0的个数
            count_ones = np.sum(self.label == 1)
            count_zeros = np.sum(self.label == 0)
            print(f"Number of 1s: {count_ones}")
            print(f"Number of 0s: {count_zeros}")
        if augment:
            self.scaler = StandardScaler()
            self.cha = self.scaler.fit_transform(self.cha)
        else:
            self.cha = self.scaler.transform(self.cha)
        

        print(f'''ecg.shape: {self.ecg.shape},
              tar.shape: {self.tar.shape}, snp.shape: {self.snp.shape}, cha.shape: {self.cha.shape}, 
              I21.shape: {len(self.I21)}, I42.shape: {len(self.I42)}, I48.shape: {len(self.I48)}, 
              I50.shape: {len(self.I50)}, I08.shape: {len(self.I08)}, I25.shape: {len(self.I25)},
              I34.shape: {len(self.I34)}, I35.shape: {len(self.I35)}, check_time.shape: {len(self.check_time)}''')
        # print(self.snp[0][0])
        # print(self.snp[0][1])
    def get_labels(self):   
        return self.label 
    def get_indicator(self):
        feature_0_17 = self.cha[:, 0:18]
        min_10_feature_0_17_idx = []
        max_10_feature_0_17_idx = []
        for i in range(18):
            feature = feature_0_17[:, i]
            min_10_feature_0_17_idx.append(np.argsort(feature)[:10])
            max_10_feature_0_17_idx.append(np.argsort(feature)[-10:])
        cha_ori = self.scaler.inverse_transform(self.cha)
        obj = {
            'min':[],
            'max':[],
            'min_eid':[],
            'max_eid':[],
            'min_ecg':[],
            'max_ecg':[],
            'min_select_tar':[],
            'max_select_tar':[],
        }
        for i in range(18):
            obj['min'].append(cha_ori[min_10_feature_0_17_idx[i]])
            obj['max'].append(cha_ori[max_10_feature_0_17_idx[i]])
            obj['min_eid'].append(self.eid[min_10_feature_0_17_idx[i]])
            obj['max_eid'].append(self.eid[max_10_feature_0_17_idx[i]])
            obj['min_ecg'].append(self.ecg[min_10_feature_0_17_idx[i]])
            obj['max_ecg'].append(self.ecg[max_10_feature_0_17_idx[i]])
            obj['min_select_tar'].append(self.select_tar[min_10_feature_0_17_idx[i]])
            obj['max_select_tar'].append(self.select_tar[max_10_feature_0_17_idx[i]])
        return obj


    def get_LVM_maxmin(self):
        feature_5 = self.cha[:, 5]
        min_10_feature_5_idx = np.argsort(feature_5)[:10]
        max_10_feature_5_idx = np.argsort(feature_5)[-10:]
        cha_ori = self.scaler.inverse_transform(self.cha)
        obj = {
            'LVM_min':cha_ori[min_10_feature_5_idx],
            'LVM_max':cha_ori[max_10_feature_5_idx],
            'LVM_min_eid': self.eid[min_10_feature_5_idx],
            'LVM_max_eid': self.eid[max_10_feature_5_idx],
            'LVM_min_ecg': self.ecg[min_10_feature_5_idx],
            'LVM_max_ecg': self.ecg[max_10_feature_5_idx],
            'LVM_min_select_tar': self.select_tar[min_10_feature_5_idx],
            'LVM_max_select_tar': self.select_tar[max_10_feature_5_idx],
        }
        return obj
    
    def get_RVEDV_maxmin(self):
        feature_5 = self.cha[:, 6]
        min_10_feature_5_idx = np.argsort(feature_5)[:10]
        max_10_feature_5_idx = np.argsort(feature_5)[-10:]
        cha_ori = self.scaler.inverse_transform(self.cha)
        obj = {
            'RVEDV_min':cha_ori[min_10_feature_5_idx],
            'RVEDV_max':cha_ori[max_10_feature_5_idx],
            'RVEDV_min_eid': self.eid[min_10_feature_5_idx],
            'RVEDV_max_eid': self.eid[max_10_feature_5_idx],
            'RVEDV_min_ecg': self.ecg[min_10_feature_5_idx],
            'RVEDV_max_ecg': self.ecg[max_10_feature_5_idx],
            'RVEDV_min_select_tar': self.select_tar[min_10_feature_5_idx],
            'RVEDV_max_select_tar': self.select_tar[max_10_feature_5_idx],
        }
        return obj
    
    def get_scaler(self):
        return self.scaler
    def add_random_noise(self, image, noise_range=(-0.1, 0.1)):
        noise = torch.FloatTensor(image.size()).uniform_(*noise_range)
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, -1, 1)  # 确保像素值在 [-1, 1] 范围内
        return noisy_image
    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.ecg)

    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
        ecg = self.ecg[index]
       
        tar = self.tar[index]
        snp = self.snp[index]
        cha = self.cha[index]
        I21 = self.I21[index]
        I42 = self.I42[index]
        I48 = self.I48[index]
        I50 = self.I50[index]
        I08 = self.I08[index]
        I25 = self.I25[index]
        I34 = self.I34[index]
        I35 = self.I35[index]
        check_time = self.check_time[index]
        select_tar = self.select_tar[index]

        if self.transform == True and self.augment == False:# val and test stage
            
            ecg_transform = transforms.Compose([
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_transform(ecg)

           
            

        if self.augment == True:# train stage
            ecg_augment = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.ecg_input_size[-1], resize=False),
                augmentations.TimeFlip(prob=self.args.timeFlip),
                augmentations.SignFlip(prob=self.args.signFlip),
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_augment(ecg)

        batch = {
                'ecg': ecg,
                'tar': tar,
                'snp': snp,
                'cha': cha,
                'I21': I21,
                'I42': I42,
                'I48': I48,
                'I50': I50,
                'select_tar': select_tar,
                'I08': I08,
                'I25': I25,
                'I34': I34,
                'I35': I35,
                'check_time': check_time
            }
        return batch


class mutimodal_dataset_CMR(data.Dataset):
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
       
        self.cmr = self.data[str(prefix+'_cmr_data')]
        self.tar = self.data[str(prefix+'_tar_data')]
        self.snp = process_snp(self.data[str(prefix+'_snp_data')])
        self.select_tar = self.data[str(prefix+'_select_tar_data')]
        self.cha = self.data[str(prefix+'_cha_data')]
        self.I21 = self.data[str(prefix+'_I21_data')]
        self.I42 = self.data[str(prefix+'_I42_data')]
        self.I48 = self.data[str(prefix+'_I48_data')]
        self.I50 = self.data[str(prefix+'_I50_data')]
        self.I08 = self.data[str(prefix+'_I08_data')]
        self.I25 = self.data[str(prefix+'_I25_data')]
        self.I34 = self.data[str(prefix+'_I34_data')]
        self.I35 = self.data[str(prefix+'_I35_data')]
        self.check_time = self.data[str(prefix+'_check_time')]
       

        if downstream == 'classification':
            if args.classification_dis == 'I21':
                self.label = np.array([1 if x != '0' else 0 for x in self.I21])
            elif args.classification_dis == 'I42':
                self.label = np.array([1 if x != '0' else 0 for x in self.I42])
            elif args.classification_dis == 'I48':
                self.label = np.array([1 if x != '0' else 0 for x in self.I48])
            elif args.classification_dis == 'I50':
                self.label = np.array([1 if x != '0' else 0 for x in self.I50])
            elif args.classification_dis == 'I08':
                self.label = np.array([1 if x != '0' else 0 for x in self.I08])
            elif args.classification_dis == 'I25':
                self.label = np.array([1 if x != '0' else 0 for x in self.I25])
            elif args.classification_dis == 'I34':
                self.label = np.array([1 if x != '0' else 0 for x in self.I34])
            elif args.classification_dis == 'I35':
                self.label = np.array([1 if x != '0' else 0 for x in self.I35])
            #统计label中1和0的个数
            count_ones = np.sum(self.label == 1)
            count_zeros = np.sum(self.label == 0)
            print(f"Number of 1s: {count_ones}")
            print(f"Number of 0s: {count_zeros}")
        if augment:
            self.scaler = StandardScaler()
            self.cha = self.scaler.fit_transform(self.cha)
        else:
            self.cha = self.scaler.transform(self.cha)
        

        print(f''' cmr.shape: {self.cmr.shape}, 
              tar.shape: {self.tar.shape}, snp.shape: {self.snp.shape}, cha.shape: {self.cha.shape}, 
              I21.shape: {len(self.I21)}, I42.shape: {len(self.I42)}, I48.shape: {len(self.I48)}, 
              I50.shape: {len(self.I50)}, I08.shape: {len(self.I08)}, I25.shape: {len(self.I25)},
              I34.shape: {len(self.I34)}, I35.shape: {len(self.I35)}, check_time.shape: {len(self.check_time)}''')
        # print(self.snp[0][0])
        # print(self.snp[0][1])
    def get_labels(self):   
        return self.label 
    def get_scaler(self):
        return self.scaler
    def add_random_noise(self, image, noise_range=(-0.1, 0.1)):
        noise = torch.FloatTensor(image.size()).uniform_(*noise_range)
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, -1, 1)  # 确保像素值在 [-1, 1] 范围内
        return noisy_image
    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.cmr)

    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
       
        cmr = self.cmr[index]
      
        tar = self.tar[index]
        snp = self.snp[index]
        cha = self.cha[index]
        I21 = self.I21[index]
        I42 = self.I42[index]
        I48 = self.I48[index]
        I50 = self.I50[index]
        I08 = self.I08[index]
        I25 = self.I25[index]
        I34 = self.I34[index]
        I35 = self.I35[index]
        check_time = self.check_time[index]
        select_tar = self.select_tar[index]

        if self.transform == True and self.augment == False:# val and test stage
            
            

           
            if self.args.resizeshape is not None:
                cmr_transform = transforms.Compose([
                    transforms.Resize(size=self.args.resizeshape),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_transform = transforms.Compose([
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            cmr = cmr_transform(cmr)
            
  
           
            

        if self.augment == True:# train stage
            

            assert cmr.shape[0] == 50 and cmr.shape[1] == 80 and cmr.shape[2] == 80
            
            if self.args.resizeshape is not None:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(size=self.args.resizeshape, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            cmr = cmr_augment(cmr)
            
         
            
        
        batch = {
           
            'cmr': cmr,
            'tar': tar,
            'snp': snp,
            'cha': cha,
            'I21': I21,
            'I42': I42,
            'I48': I48,
            'I50': I50,
            'select_tar': select_tar,
            
            'I08': I08,
            'I25': I25,
            'I34': I34,
            'I35': I35,
            'check_time': check_time
        }
        
        return batch
    


class mutimodal_dataset_laCMR(data.Dataset):
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
        
      
        self.tar = self.data[str(prefix+'_tar_data')]
        self.snp = process_snp(self.data[str(prefix+'_snp_data')])
        self.select_tar = self.data[str(prefix+'_select_tar_data')]
        self.cha = self.data[str(prefix+'_cha_data')]
        self.I21 = self.data[str(prefix+'_I21_data')]
        self.I42 = self.data[str(prefix+'_I42_data')]
        self.I48 = self.data[str(prefix+'_I48_data')]
        self.I50 = self.data[str(prefix+'_I50_data')]
        self.I08 = self.data[str(prefix+'_I08_data')]
        self.I25 = self.data[str(prefix+'_I25_data')]
        self.I34 = self.data[str(prefix+'_I34_data')]
        self.I35 = self.data[str(prefix+'_I35_data')]
        self.check_time = self.data[str(prefix+'_check_time')]
        self.la_cmr = self.data[str(prefix+'_la_cmr_data')]

        if downstream == 'classification':
            if args.classification_dis == 'I21':
                self.label = np.array([1 if x != '0' else 0 for x in self.I21])
            elif args.classification_dis == 'I42':
                self.label = np.array([1 if x != '0' else 0 for x in self.I42])
            elif args.classification_dis == 'I48':
                self.label = np.array([1 if x != '0' else 0 for x in self.I48])
            elif args.classification_dis == 'I50':
                self.label = np.array([1 if x != '0' else 0 for x in self.I50])
            elif args.classification_dis == 'I08':
                self.label = np.array([1 if x != '0' else 0 for x in self.I08])
            elif args.classification_dis == 'I25':
                self.label = np.array([1 if x != '0' else 0 for x in self.I25])
            elif args.classification_dis == 'I34':
                self.label = np.array([1 if x != '0' else 0 for x in self.I34])
            elif args.classification_dis == 'I35':
                self.label = np.array([1 if x != '0' else 0 for x in self.I35])
            #统计label中1和0的个数
            count_ones = np.sum(self.label == 1)
            count_zeros = np.sum(self.label == 0)
            print(f'classification_dis:{args.classification_dis}')
            print(f"Number of 1s: {count_ones}")
            print(f"Number of 0s: {count_zeros}")
        if augment:
            self.scaler = StandardScaler()
            self.cha = self.scaler.fit_transform(self.cha)
        else:
            self.cha = self.scaler.transform(self.cha)
        

        print(f'''la_cmr.shape: {len(self.la_cmr)}, 
              tar.shape: {self.tar.shape}, snp.shape: {self.snp.shape}, cha.shape: {self.cha.shape}, 
              I21.shape: {len(self.I21)}, I42.shape: {len(self.I42)}, I48.shape: {len(self.I48)}, 
              I50.shape: {len(self.I50)}, I08.shape: {len(self.I08)}, I25.shape: {len(self.I25)},
              I34.shape: {len(self.I34)}, I35.shape: {len(self.I35)}, check_time.shape: {len(self.check_time)}''')
        # print(self.snp[0][0])
        # print(self.snp[0][1])
    def get_labels(self):   
        return self.label 
    def get_scaler(self):
        return self.scaler
    def add_random_noise(self, image, noise_range=(-0.1, 0.1)):
        noise = torch.FloatTensor(image.size()).uniform_(*noise_range)
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, -1, 1)  # 确保像素值在 [-1, 1] 范围内
        return noisy_image
    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.la_cmr)

    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
        
        la_cmr = torch.from_numpy(self.la_cmr[index]).float()
        tar = self.tar[index]
        snp = self.snp[index]
        cha = self.cha[index]
        I21 = self.I21[index]
        I42 = self.I42[index]
        I48 = self.I48[index]
        I50 = self.I50[index]
        I08 = self.I08[index]
        I25 = self.I25[index]
        I34 = self.I34[index]
        I35 = self.I35[index]
        check_time = self.check_time[index]
        select_tar = self.select_tar[index]

        if self.transform == True and self.augment == False:# val and test stage
            
           

            la_cmr = (la_cmr-la_cmr.min())/(la_cmr.max()-la_cmr.min())
            if self.args.resizeshape is not None:
                cmr_transform = transforms.Compose([
                    transforms.Resize(size=self.args.resizeshape),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_transform = transforms.Compose([
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            
            la_cmr = cmr_transform(la_cmr)
            
            

        if self.augment == True:# train stage
            

           
            assert la_cmr.shape[0] == 50 and la_cmr.shape[1] == 96 and la_cmr.shape[2] == 96
            la_cmr = (la_cmr-la_cmr.min())/(la_cmr.max()-la_cmr.min())
            if self.args.resizeshape is not None:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(size=self.args.resizeshape, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            
            la_cmr = cmr_augment(la_cmr)
            
        batch = {
            'tar': tar,
            'snp': snp,
            'cha': cha,
            'I21': I21,
            'I42': I42,
            'I48': I48,
            'I50': I50,
            'select_tar': select_tar,
            'la_cmr': la_cmr,
            'I08': I08,
            'I25': I25,
            'I34': I34,
            'I35': I35,
            'check_time': check_time
        }
        return batch

class mutimodal_dataset_zheyi_ECG(data.Dataset):
    def __init__(self,data_path=None,
                transform=True,
                augment=False,
                args=None,
                scaler=None,
                downstream=None,# downstream task, include classification and regression
                 ):
        super().__init__()
        self.downstream = downstream
        self.transform=transform
        self.augment=augment
        self.args=args
        self.data = pickle.load(open(data_path, 'rb'))
        if 'binglihao' in self.data:
            self.binglihao = self.data['binglihao']
        elif 'path' in self.data:
            self.binglihao = self.data['path']
        elif 'eid' in self.data:
            self.binglihao = self.data['eid']
        else:
            print('no binglihao or path')
        self.scaler = scaler
        if 'renji' in data_path and downstream == 'regression':
            self.select_tar = self.data['select_tar']
        else:
            self.select_tar = np.array([np.array([self.data['sex'][i],self.data['age'][i],self.data['heartbeat'][i]]) for i in range(len(self.data['age']))])
        self.I42 = np.array(self.data['label'])
        self.ecg = torch.from_numpy(np.array(self.data['ecg'])).float()
        if torch.isnan(self.ecg).any():
            print("Warning: ecg contains NaN")
            exit()
        if 'renji' in data_path:
            print('no need for label process')
            self.label = self.I42
        else:
            if downstream == 'classification':
                if args.classification_dis == 'I42':
                    self.label = np.array([1 if x != 0 else 0 for x in self.I42])
                #统计label中1和0的个数
                count_ones = np.sum(self.label == 1)
                count_zeros = np.sum(self.label == 0)
                print(f"Number of 1s: {count_ones}")
                print(f"Number of 0s: {count_zeros}")
            elif downstream == 'yaxing':
                non_zero_indices = [index for index, value in enumerate(self.I42) if value != 0]
                self.label = np.array([self.I42[index] for index in non_zero_indices])
                self.ecg = self.ecg[non_zero_indices]
                self.select_tar = self.select_tar[non_zero_indices]
                print(f"Number of non-zero values: {len(non_zero_indices)}")
                self.label[self.label == 3] = 0
                count_ones = np.sum(self.label == 1)
                count_twos = np.sum(self.label == 2)
                count_threes = np.sum(self.label == 0)
                print(f'Number of 1s: {count_ones}, Number of 2s: {count_twos}, Number of 3s: {count_threes}')
        
        if 'renji' in data_path and downstream == 'regression':
            print('no standar renkouxue zhibiao')
            pass
        else:
            print('standar renkouxue zhibiao')
            if augment:
                self.scaler = StandardScaler()
                self.select_tar = self.scaler.fit_transform(self.select_tar)
            else:
                self.select_tar = self.scaler.transform(self.select_tar)
        del self.data
        print(f'''ecg.shape: {self.ecg.shape},ecg.dtype:{self.ecg.dtype},
        label.shape: {self.label.shape}, label.dtype:{self.label.dtype},
        select_tar.shape: {self.select_tar.shape}, select_tar.dtype:{self.select_tar.dtype}''')


    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.ecg)
    def get_scaler(self):
        return self.scaler
    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
        ecg = self.ecg[index]
        select_tar = self.select_tar[index]
        I42 = self.label[index]
        if torch.isnan(ecg).any():
            print("Warning:3333 ecg contains NaN")
            exit()
        if self.transform == True and self.augment == False:# val and test stage
            
            ecg_transform = transforms.Compose([
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_transform(ecg)
        if torch.isnan(ecg).any():
            print("Warning:22222 ecg contains NaN")
            exit()

        if self.augment == True:# train stage
            ecg_augment = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.ecg_input_size[-1], resize=False),
                augmentations.TimeFlip(prob=self.args.timeFlip),
                augmentations.SignFlip(prob=self.args.signFlip),
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_augment(ecg)
        # print(f'ecg.dtype:{ecg.dtype},select_tar.dtype:{select_tar.dtype},I42.dtype:{I42.dtype}')
        batch = {
            'ecg': ecg,
            'select_tar': select_tar,
            'I42': I42,
            'cha': I42,
            'binglihao':self.binglihao[index]
        }
        return batch

class mutimodal_dataset_MIMIC_ECG(data.Dataset):
    def __init__(self,data_path=None,
                transform=True,
                augment=False,
                args=None,
                scaler=None,
                downstream=None,# downstream task, include classification and regression
                 ):
        super().__init__()
        self.downstream = downstream
        self.transform=transform
        self.augment=augment
        self.args=args
        data = pickle.load(open(data_path, 'rb'))
        
        if downstream == 'classification':
            label_index = [index for index, value in enumerate(data[args.classification_dis]) if value == 1]
            yingxing_index = [index for index, value in enumerate(data['yingxing']) if value == 1]
            self.select_tar = np.concatenate((data['select_tar'][label_index], data['select_tar'][yingxing_index]), axis=0)
            valid_indices = np.where(~np.isnan(self.select_tar).any(axis=1))[0]

            self.ecg = np.concatenate((data['ecg'][label_index], data['ecg'][yingxing_index]), axis=0)
            self.label = np.array([1]*len(label_index) + [0]*len(yingxing_index))
            self.ecg = torch.from_numpy(self.ecg).float()

            self.ecg = self.ecg[valid_indices]
            self.select_tar = self.select_tar[valid_indices]
            self.label = self.label[valid_indices]
            
            if torch.isnan(self.ecg).any():
                print("Warning: ecg contains NaN")
                exit()
            count_ones = np.sum(self.label == 1)
            count_zeros = np.sum(self.label == 0)
            print(f"Number of 1s: {count_ones}")
            print(f"Number of 0s: {count_zeros}")
            del data
        elif downstream == 'yaxing':
            kuozhang_index = [index for index, value in enumerate(data['kuozhang']) if value == 1]
            feihou_index = [index for index, value in enumerate(data['feihou']) if value == 1]
            xianzhixing = [index for index, value in enumerate(data['xianzhixing']) if value == 1]
            self.select_tar = np.concatenate((data['select_tar'][kuozhang_index], data['select_tar'][feihou_index], data['select_tar'][xianzhixing]), axis=0)
            self.ecg = np.concatenate((data['ecg'][kuozhang_index], data['ecg'][feihou_index], data['ecg'][xianzhixing]), axis=0)
            self.label = np.array([1]*len(kuozhang_index) + [2]*len(feihou_index) + [0]*len(xianzhixing))
            self.ecg = torch.from_numpy(self.ecg).float()
            if torch.isnan(self.ecg).any():
                print("Warning: ecg contains NaN")
                exit()
            count_ones = np.sum(self.label == 1)
            count_twos = np.sum(self.label == 2)
            count_threes = np.sum(self.label == 0)
            print(f'Number of 1s: {count_ones}, Number of 2s: {count_twos}, Number of 3s: {count_threes}')
            del data

        print(f'''ecg.shape: {self.ecg.shape},ecg.dtype:{self.ecg.dtype},
        label.shape: {self.label.shape}, label.dtype:{self.label.dtype},
        select_tar.shape: {self.select_tar.shape}, select_tar.dtype:{self.select_tar.dtype}''')


    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.ecg)

    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
        ecg = self.ecg[index]
        select_tar = self.select_tar[index]
        I42 = self.label[index]
        if torch.isnan(ecg).any():
            print("Warning:3333 ecg contains NaN")
            exit()
        if self.transform == True and self.augment == False:# val and test stage
            
            ecg_transform = transforms.Compose([
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_transform(ecg)
        if torch.isnan(ecg).any():
            print("Warning:22222 ecg contains NaN")
            exit()

        if self.augment == True:# train stage
            ecg_augment = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.ecg_input_size[-1], resize=False),
                augmentations.TimeFlip(prob=self.args.timeFlip),
                augmentations.SignFlip(prob=self.args.signFlip),
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_augment(ecg)
        # print(f'ecg.dtype:{ecg.dtype},select_tar.dtype:{select_tar.dtype},I42.dtype:{I42.dtype}')
        batch = {
            'ecg': ecg,
            'select_tar': select_tar,
            'I42': I42,
            'cha': I42
        }
        return batch




class mutimodal_dataset_NEWECG(data.Dataset):
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
        self.data = pickle.load(open(data_path, 'rb'))
        
        prefix = data_path.split('/')[-1].split('_')[0]
        if prefix == 'trainval':
            prefix = 'train'
            
        if prefix == 'train':
            self.healthy_data = pickle.load(open('/mnt/data2/ECG_CMR/UKB_select_data/train_data_v11_healthy.pt', 'rb'))
        else:
            self.healthy_data = pickle.load(open('/mnt/data2/ECG_CMR/UKB_select_data/test_data_v11_healthy.pt', 'rb'))

        self.ecg = self.data[str(prefix+'_ecg_data')]
        self.select_tar = self.data[str(prefix+'_select_tar_data')]
        self.label = torch.from_numpy(self.data[str(prefix+'_label')])
        self.cmr = self.data[str(prefix+'_cmr_data')]
        self.la_cmr = torch.from_numpy(self.data[str(prefix+'_la_cmr_data')])
        
        self.health_ecg = self.healthy_data[str(prefix+'_ecg_data')][:len(self.ecg)]
        self.health_select_tar = self.healthy_data[str(prefix+'_select_tar_data')][:len(self.ecg)]
        self.health_label = torch.from_numpy(self.healthy_data[str(prefix+'_label')][:len(self.ecg)])
        self.health_cmr = self.healthy_data[str(prefix+'_cmr_data')][:len(self.ecg)]
        self.health_la_cmr = torch.from_numpy(self.healthy_data[str(prefix+'_la_cmr_data')][:len(self.ecg)])
        

        self.label = torch.from_numpy(np.array([1]*len(self.ecg) + [0]*len(self.health_ecg)))
        self.ecg = torch.cat((self.ecg, self.health_ecg), dim=0)
        self.select_tar = torch.cat((self.select_tar, self.health_select_tar), dim=0)
        self.cmr = torch.cat((self.cmr, self.health_cmr), dim=0)
        self.la_cmr = torch.cat((self.la_cmr, self.health_la_cmr), dim=0)
        
        del self.health_ecg, self.health_select_tar, self.health_label, self.health_cmr, self.health_la_cmr
        del self.healthy_data
        del self.data
        

        print(f'''ecg.shape: {self.ecg.shape},ecg.dtype:{self.ecg.dtype}, select_tar.shape: {self.select_tar.shape}, select_tar.dtype:{self.select_tar.dtype}, label.shape: {self.label.shape}, label.dtype:{self.label.dtype},
        cmr.shape: {self.cmr.shape}, cmr.dtype:{self.cmr.dtype}, la_cmr.shape: {self.la_cmr.shape}, la_cmr.dtype:{self.la_cmr.dtype}''')
    
    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.ecg)

    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
        ecg = self.ecg[index]
        select_tar = self.select_tar[index]
        cmr = self.cmr[index]
        la_cmr = self.la_cmr[index]

        if self.transform == True and self.augment == False:# val and test stage
            
            ecg_transform = transforms.Compose([
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_transform(ecg)

            la_cmr = (la_cmr-la_cmr.min())/(la_cmr.max()-la_cmr.min())
            if self.args.resizeshape is not None:
                cmr_transform = transforms.Compose([
                    transforms.Resize(size=self.args.resizeshape),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_transform = transforms.Compose([
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            cmr = cmr_transform(cmr)
            la_cmr = cmr_transform(la_cmr)

        if self.augment == True:# train stage
            ecg_augment = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.ecg_input_size[-1], resize=False),
                augmentations.TimeFlip(prob=self.args.timeFlip),
                augmentations.SignFlip(prob=self.args.signFlip),
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_augment(ecg)

            assert cmr.shape[0] == 50 and cmr.shape[1] == 80 and cmr.shape[2] == 80
            assert la_cmr.shape[0] == 50 and la_cmr.shape[1] == 96 and la_cmr.shape[2] == 96
            la_cmr = (la_cmr-la_cmr.min())/(la_cmr.max()-la_cmr.min())
            if self.args.resizeshape is not None:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(size=self.args.resizeshape, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            cmr = cmr_augment(cmr)
            la_cmr = cmr_augment(la_cmr)
        batch = {
                'ecg': ecg,
                'select_tar': select_tar,
                'cha': self.label[index],
                'cmr': cmr,
                'la_cmr': la_cmr
            }
        return batch



class mutimodal_dataset_MIMIC_NEWECG(data.Dataset):
    def __init__(self,data_path=None,
                transform=True,
                augment=False,
                args=None,
                scaler=None,
                downstream=None,# downstream task, include classification and regression
                 ):
        super().__init__()
        self.downstream = downstream
        self.transform=transform
        self.augment=augment
        self.args=args
        self.scaler = scaler
        data = pickle.load(open(data_path, 'rb'))
        prefix = data_path.split('/')[-1].split('_')[-1].split('.')[0]
        print(f'prefix:{prefix}')
        if prefix == 'train':
            health = pickle.load(open('/mnt/data2/ECG_CMR/mimic_data/mimic-iv-ecg-ext-icd-diagnostic-labels-for-mimic-iv-ecg-1.0.0/Sametarprocess/random_train.pkl', 'rb'))
        elif prefix == 'test':
            health = pickle.load(open('/mnt/data2/ECG_CMR/mimic_data/mimic-iv-ecg-ext-icd-diagnostic-labels-for-mimic-iv-ecg-1.0.0/Sametarprocess/random_test.pkl', 'rb'))
        else:
            print('prefix error')
            exit()
        
        label1_eid = np.array(data['eid'])
        label0_eid = np.array(health['eid'])
        print(f'label1_eid.shape:{label1_eid.shape},label0_eid.shape:{label0_eid.shape}')
        print("label1_eid 唯一值的数量:", len(np.unique(label1_eid)))
        print("label0_eid 唯一值的数量:", len(np.unique(label0_eid)))

        mask = ~np.isin(label0_eid, label1_eid)
        indices = np.where(mask)[0].tolist()
        print(f'len(indices):{len(indices)}')
        print(type(indices))
        print(type(health['select_tar']))
        health['ecg'] = health['ecg'][indices]
        health['select_tar'] = np.array(health['select_tar'])[indices]
        




        self.ecg = data['ecg']
        self.select_tar = data['select_tar']
        self.label = data['label']

        self.health_ecg = health['ecg'][0:len(self.ecg)]
        self.health_select_tar = health['select_tar'][0:len(self.ecg)]
        self.health_label = np.array([0]*len(self.ecg))
        
        self.ecg = np.concatenate((self.ecg, self.health_ecg), axis=0)
        self.select_tar = np.concatenate((self.select_tar, self.health_select_tar), axis=0)
        self.label = np.concatenate((self.label, self.health_label), axis=0)
        valid_indices = np.where(~np.isnan(self.select_tar).any(axis=1))[0]

        self.ecg = self.ecg[valid_indices]
        self.select_tar = self.select_tar[valid_indices]
        self.label = self.label[valid_indices]
        
        if augment:
            self.scaler = StandardScaler()
            self.select_tar = self.scaler.fit_transform(self.select_tar)
        else:
            self.select_tar = self.scaler.transform(self.select_tar)
        # 打印label中0和1的数量
        count_ones = np.sum(self.label == 1).item()
        count_zeros = np.sum(self.label == 0).item()

        self.ecg = torch.from_numpy(self.ecg).float()
        self.select_tar = torch.from_numpy(self.select_tar).float()
        self.label = torch.from_numpy(self.label).float()
        if prefix == 'train':
            if args.training_percentage == 0.1:
                print(f'args.training_percentage:{args.training_percentage}')
                num_samples = self.ecg.shape[0] // 10
                indices = torch.randperm(self.ecg.shape[0])[:num_samples]
                self.ecg = self.ecg[indices]
                self.select_tar = self.select_tar[indices]
                self.label = self.label[indices]
            elif args.training_percentage == 0.25:
                print(f'args.training_percentage:{args.training_percentage}')
                num_samples = self.ecg.shape[0] // 4
                indices = torch.randperm(self.ecg.shape[0])[:num_samples]
                self.ecg = self.ecg[indices]
                self.select_tar = self.select_tar[indices]
                self.label = self.label[indices]
            elif args.training_percentage == 0.5:
                num_samples = self.ecg.shape[0] // 2
                indices = torch.randperm(self.ecg.shape[0])[:num_samples]
                self.ecg = self.ecg[indices]
                self.select_tar = self.select_tar[indices]
                self.label = self.label[indices]
            elif args.training_percentage == 0.75:
                num_samples = int(self.ecg.shape[0] * 0.75)
                indices = torch.randperm(self.ecg.shape[0])[:num_samples]
                self.ecg = self.ecg[indices]
                self.select_tar = self.select_tar[indices]
                self.label = self.label[indices]
                print(f'args.training_percentage:{args.training_percentage}')
            elif args.training_percentage == 1.0:
                print(f'args.training_percentage:{args.training_percentage}')
            else:
                print(f'error:args.training_percentage:{args.training_percentage}')
                exit()


        print(f"Number of 1s: {count_ones}")   
        print(f"Number of 0s: {count_zeros}")
        
        print(self.label)
        del self.health_ecg, self.health_select_tar, self.health_label
        del health
        del data
        if np.isnan(self.ecg).any():
            print("Warning: ecg contains NaN")
            exit()
        if np.isnan(self.select_tar).any():
            print("Warning: cond contains NaN")
            exit()
        print(f'''ecg.shape: {self.ecg.shape},ecg.dtype:{self.ecg.dtype}, select_tar.shape: {self.select_tar.shape}, select_tar.dtype:{self.select_tar.dtype}, label.shape: {self.label.shape}, label.dtype:{self.label.dtype}''')


    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.ecg)
    def get_scaler(self):
        return self.scaler
    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
        ecg = self.ecg[index]
        select_tar = self.select_tar[index]
        cha = self.label[index]
        
        if self.transform == True and self.augment == False:# val and test stage
            
            ecg_transform = transforms.Compose([
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_transform(ecg)
        

        if self.augment == True:# train stage
            ecg_augment = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.ecg_input_size[-1], resize=False),
                augmentations.TimeFlip(prob=self.args.timeFlip),
                augmentations.SignFlip(prob=self.args.signFlip),
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_augment(ecg)
        # print(f'ecg.dtype:{ecg.dtype},select_tar.dtype:{select_tar.dtype},I42.dtype:{I42.dtype}')
        batch = {
            'ecg': ecg,
            'select_tar': select_tar,
            'cha': cha
        }
        return batch




class mutimodal_dataset_Gen_CMRlaCMR(data.Dataset):
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

        if args.cmr_isreal:
            self.sa_cmr = self.data['sa_cmr_data']
            self.la_cmr = self.data['la_cmr_data']
        elif args.cmr_isreal == False:
            self.sa_cmr = self.data['fake_sa_cmr_data']
            self.la_cmr = self.data['fake_la_cmr_data']
        else:
            print('error:args.cmr_isreal')
            exit()
        self.cha = self.data['cha']
        self.select_tar = self.data['select_tar']


        if downstream == 'classification':
            print(f'downstream: error')
            exit()

        if augment:
            self.scaler = StandardScaler()
            self.cha = self.scaler.fit_transform(self.cha)
        else:
            self.cha = self.scaler.transform(self.cha)
        

        print(f''' cmr.shape: {self.sa_cmr.shape},la_cmr.shape: {len(self.la_cmr.shape)}, 
               cha.shape: {self.cha.shape}, select_tar.shape: {self.select_tar.shape}''')
    
    def get_scaler(self):
        return self.scaler
    
    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.sa_cmr)

    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
       
        cmr = torch.from_numpy(self.sa_cmr[index]).float()
        la_cmr = torch.from_numpy(self.la_cmr[index]).float()
        select_tar = self.select_tar[index]
        cha = self.cha[index]

        if self.transform == True and self.augment == False:# val and test stage

            la_cmr = (la_cmr-la_cmr.min())/(la_cmr.max()-la_cmr.min())
            if self.args.resizeshape is not None:
                cmr_transform = transforms.Compose([
                    transforms.Resize(size=self.args.resizeshape),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_transform = transforms.Compose([
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            cmr = cmr_transform(cmr)
            la_cmr = cmr_transform(la_cmr)

        if self.augment == True:# train stage
            
            assert cmr.shape[0] == 50 and cmr.shape[1] == 80 and cmr.shape[2] == 80
            assert la_cmr.shape[0] == 50 and la_cmr.shape[1] == 96 and la_cmr.shape[2] == 96
            la_cmr = (la_cmr-la_cmr.min())/(la_cmr.max()-la_cmr.min())
            if self.args.resizeshape is not None:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(size=self.args.resizeshape, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            else:
                cmr_augment = transforms.Compose([
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            cmr = cmr_augment(cmr)
            la_cmr = cmr_augment(la_cmr)
        
        batch = {
                'cmr': cmr,
                'la_cmr': la_cmr,
                'select_tar': select_tar,
                'cha': cha
            }
        return batch

def get_train_dataset_class(name,args):
    if name == 'mutimodal_dataset':
        return mutimodal_dataset(data_path=args.train_data_path, transform=True, augment=True, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_ECGCMR':
        return mutimodal_dataset_ECGCMR(data_path=args.train_data_path, transform=True, augment=True, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_ECGlaCMR':
        return mutimodal_dataset_ECGlaCMR(data_path=args.train_data_path, transform=True, augment=True, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_CMRlaCMR':
        return mutimodal_dataset_CMRlaCMR(data_path=args.train_data_path, transform=True, augment=True, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_ECG':
        return mutimodal_dataset_ECG(data_path=args.train_data_path, transform=True, augment=True, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_CMR':
        return mutimodal_dataset_CMR(data_path=args.train_data_path, transform=True, augment=True, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_laCMR':
        return mutimodal_dataset_laCMR(data_path=args.train_data_path, transform=True, augment=True, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_zheyi_ECG':
        return mutimodal_dataset_zheyi_ECG(data_path=args.train_data_path, transform=True, augment=True, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_MIMIC_ECG':
        return mutimodal_dataset_MIMIC_ECG(data_path=args.train_data_path, transform=True, augment=True, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_NEWECG':
        return mutimodal_dataset_NEWECG(data_path=args.train_data_path, transform=True, augment=True, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_MIMIC_NEWECG':
        return mutimodal_dataset_MIMIC_NEWECG(data_path=args.train_data_path, transform=True, augment=True, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_Gen_CMRlaCMR':
        return mutimodal_dataset_Gen_CMRlaCMR(data_path=args.train_data_path, transform=True, augment=True, args=args,downstream=args.downstream)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
def get_test_dataset_class(name,args,scaler=None):
    if name == 'mutimodal_dataset':
        return mutimodal_dataset(data_path=args.test_data_path, scaler=scaler, transform=True, augment=False, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_ECGCMR':
        return mutimodal_dataset_ECGCMR(data_path=args.test_data_path, scaler=scaler,transform=True, augment=False, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_ECGlaCMR':
        return mutimodal_dataset_ECGlaCMR(data_path=args.test_data_path, scaler=scaler,transform=True, augment=False, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_CMRlaCMR':
        return mutimodal_dataset_CMRlaCMR(data_path=args.test_data_path, scaler=scaler,transform=True, augment=False, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_ECG':
        return mutimodal_dataset_ECG(data_path=args.test_data_path,scaler=scaler, transform=True, augment=False, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_CMR':
        return mutimodal_dataset_CMR(data_path=args.test_data_path, scaler=scaler,transform=True, augment=False, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_laCMR':
        return mutimodal_dataset_laCMR(data_path=args.test_data_path, scaler=scaler,transform=True, augment=False, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_zheyi_ECG':
        return mutimodal_dataset_zheyi_ECG(data_path=args.test_data_path, scaler=scaler,transform=True, augment=False, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_MIMIC_ECG':
        return mutimodal_dataset_MIMIC_ECG(data_path=args.test_data_path, scaler=None,transform=True, augment=False, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_NEWECG':
        return mutimodal_dataset_NEWECG(data_path=args.test_data_path, scaler=None,transform=True, augment=False, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_MIMIC_NEWECG':
        return mutimodal_dataset_MIMIC_NEWECG(data_path=args.test_data_path, scaler=scaler,transform=True, augment=False, args=args,downstream=args.downstream)
    elif name == 'mutimodal_dataset_Gen_CMRlaCMR':
        return mutimodal_dataset_Gen_CMRlaCMR(data_path=args.test_data_path, scaler=scaler,transform=True, augment=False, args=args,downstream=args.downstream)
    else:
        raise ValueError(f"Unknown dataset: {name}")


if  __name__ == '__main__':
    import argparse
    
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Your description here')

    # 添加参数
    parser.add_argument('--input_size', type=tuple, default=(12, 5000))
    parser.add_argument('--timeFlip', type=float, default=0.33)
    parser.add_argument('--signFlip', type=float, default=0.33)
    parser.add_argument('--classification_dis', type=str, default='I42')
    parser.add_argument('--resizeshape', type=tuple, default=(256, 256))

    # 解析参数
    args = parser.parse_args()
    dataset = mutimodal_dataset_MIMIC_NEWECG(data_path='/mnt/data2/ECG_CMR/mimic_data/mimic-iv-ecg-ext-icd-diagnostic-labels-for-mimic-iv-ecg-1.0.0/MC_test.pkl',transform=True,augment=False,args=args,scaler=None,downstream='classification')
    batch = dataset[0]
    print(f"ecg:{batch['ecg'].shape},dtype:{batch['ecg'].dtype}")
    print(f"cond:{batch['select_tar'].shape},dtype:{batch['select_tar'].dtype}")
    print(f"cha:{batch['cha'].shape},dtype:{batch['cha'].dtype}")
    # print(f"cmr:{batch['cmr'].shape},dtype:{batch['cmr'].dtype}")
    # print(f"la_cmr:{batch['la_cmr'].shape},dtype:{batch['la_cmr'].dtype}")