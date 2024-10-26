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
from utils.registry_class import ECGCMRDATASET,ECGCMRDATASET_ECGCMR,ECGCMRDATASET_ECGlaCMR,ECGCMRDATASET_ECGlaCMRnew,ECGCMRDATASET_UKB_SMALL,ECGCMRDATASET_zheyi
import pickle
@ECGCMRDATASET.register_class()
class ECGCMRDATASET(data.Dataset):
    def __init__(self,data_path=None,
                transform=True,
                augment=True,
                args=None,
                scaler=None,
                downstream=None,# downstream task, include classification and regression
                resizeshape=80,
                 ):
        super().__init__()
        self.resizeshape = resizeshape
        self.downstream = downstream
        self.scaler = scaler
        self.transform=transform
        self.augment=augment
        self.args=args
        self.data = torch.load(data_path,map_location='cpu')
        prefix = data_path.split('/')[-1].split('_')[0]
        self.ecg = self.data[str(prefix+'_ecg_data')]
        self.cmr = self.data[str(prefix+'_cmr_data')]
        self.tar = self.data[str(prefix+'_tar_data')]
        self.select_tar = self.data[str(prefix+'_select_tar_data')]
        self.snp = process_snp(self.data[str(prefix+'_snp_data')])
        #
        # self.snp = torch.randint(0,2,(self.snp.shape[0],self.snp.shape[1],self.snp.shape[2]))
        #
        self.cha = self.data[str(prefix+'_cha_data')]
        self.I21 = self.data[str(prefix+'_I21_data')]
        self.I42 = self.data[str(prefix+'_I42_data')]
        self.I48 = self.data[str(prefix+'_I48_data')]
        self.I50 = self.data[str(prefix+'_I50_data')]

        if augment:
            self.scaler = StandardScaler()
            self.cha = self.scaler.fit_transform(self.cha)
        else:
            self.cha = self.scaler.transform(self.cha)
        

        print(f'ecg.shape: {self.ecg.shape}, cmr.shape: {self.cmr.shape}, tar.shape: {self.tar.shape}, snp.shape: {self.snp.shape}, cha.shape: {self.cha.shape}, I21.shape: {self.I21.shape}, I42.shape: {self.I42.shape}, I48.shape: {self.I48.shape}, I50.shape: {self.I50.shape}')
        # print(self.snp[0][0])
        # print(self.snp[0][1])
    def get_scaler(self):
        return self.scaler
    
    def save_data(self,generate_cmr,save_path):
        self.data['generate_cmr'] = generate_cmr
        torch.save(self.data,save_path,pickle_protocol=4)
        
    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.ecg)

    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
        ecg = self.ecg[index]
        cmr = self.cmr[index].unsqueeze(1).repeat(1,3,1,1)
        tar = self.tar[index]
        select_tar = self.select_tar[index]
        snp = self.snp[index]
        cha = self.cha[index]
        I21 = self.I21[index]
        I42 = self.I42[index]
        I48 = self.I48[index]
        I50 = self.I50[index]

        
        ecg_transform = transforms.Compose([
            transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
        ])
        ecg = ecg_transform(ecg)
        
        cmr_transform = transforms.Compose([
            transforms.Resize(size=self.resizeshape, antialias=True),
        ])
        cmr = cmr_transform(cmr)
        # cmr = 2 * cmr - 1
        
        
        return index,ecg,cmr,tar,snp,cha,I21,I42,I48,I50,select_tar


@ECGCMRDATASET_ECGCMR.register_class()
class mutimodal_dataset_ECGCMR(data.Dataset):
    def __init__(self,data_path=None,
                transform=True,
                augment=False,
                args=None,
                scaler=None,
                downstream=None,# downstream task, include classification and regression
                resizeshape=80,
                ):
        super().__init__()
        self.resizeshape = resizeshape
        self.downstream = downstream
        self.scaler = scaler
        self.transform=transform
        self.augment=augment
        self.args=args
        self.data = torch.load(data_path,map_location='cpu')
        prefix = data_path.split('/')[-1].split('_')[0]
        if prefix == 'trainval':
            prefix = 'train'
        self.eid = self.data[str(prefix+'_eid')]
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
        eid = self.eid[index]
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

            
            if self.resizeshape is not None:
                cmr_transform = transforms.Compose([
                    transforms.Resize(size=self.resizeshape),
                    # transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            # else:
            #     cmr_transform = transforms.Compose([
            #         transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
            #     ])
            cmr = cmr_transform(cmr)
            cmr = cmr.unsqueeze(1).repeat(1,3,1,1)
                   
            

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
            'check_time': check_time,
            'eid': eid
        }
         
        return batch
    

@ECGCMRDATASET_ECGlaCMR.register_class()
class mutimodal_dataset_ECGlaCMR(data.Dataset):
    def __init__(self,data_path=None,
                transform=True,
                augment=False,
                args=None,
                scaler=None,
                downstream=None,# downstream task, include classification and regression
                resizeshape=80,
                ):
        super().__init__()
        self.resizeshape = resizeshape
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
        eid = self.eid[index]

        if self.transform == True and self.augment == False:# val and test stage
            
            ecg_transform = transforms.Compose([
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_transform(ecg)

            la_cmr = (la_cmr-la_cmr.min())/(la_cmr.max()-la_cmr.min())
            if self.resizeshape is not None:
                cmr_transform = transforms.Compose([
                    transforms.Resize(size=self.resizeshape),
                    # transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            # else:
            #     cmr_transform = transforms.Compose([
            #         transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
            #     ])
           
            la_cmr = cmr_transform(la_cmr)
            la_cmr = la_cmr.unsqueeze(1).repeat(1,3,1,1)
            

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
                'check_time': check_time,
                'eid': eid
            }
        return batch
    




@ECGCMRDATASET_ECGlaCMRnew.register_class()
class mutimodal_dataset_ECGlaCMRNew(data.Dataset):
    def __init__(self,data_path=None,
                transform=True,
                augment=False,
                args=None,
                scaler=None,
                downstream=None,# downstream task, include classification and regression
                resizeshape=80,
                ):
        super().__init__()
        self.resizeshape = resizeshape
        
        self.transform=transform
        self.augment=augment
        self.args=args
        self.data = torch.load(data_path,map_location='cpu')
        prefix = data_path.split('/')[-1].split('_')[0]
        if prefix == 'trainval':
            prefix = 'train'
        self.ecg = self.data[str(prefix+'_ecg_data')]
        self.eid = self.data[str(prefix+'_eid')]
        self.la_cmr = self.data[str(prefix+'_la_cmr_data')]

        
        

        print(f'''ecg.shape: {self.ecg.shape}, la_cmr.shape: {len(self.la_cmr)}''')
    
    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.ecg)

    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
        ecg = self.ecg[index]
        la_cmr = self.la_cmr[index]
        eid = self.eid[index]

        if self.transform == True and self.augment == False:# val and test stage
            
            ecg_transform = transforms.Compose([
                transformations.MinMaxScaling(lower=-1, upper=1, mode="channel_wise"),
            ])
            ecg = ecg_transform(ecg)

            if self.resizeshape is not None:
                cmr_transform = transforms.Compose([
                    transforms.Resize(size=self.resizeshape),
                ])
           
            la_cmr = cmr_transform(la_cmr)
            la_cmr = la_cmr.unsqueeze(1).repeat(1,3,1,1)
            
            
        batch = {
                'ecg': ecg,
                'la_cmr': la_cmr,
                'eid': eid
            }
        return batch
    

@ECGCMRDATASET_UKB_SMALL.register_class()
class mutimodal_dataset_NEWECG(data.Dataset):
    def __init__(self,data_path=None,
                transform=True,
                augment=False,
                args=None,
                scaler=None,
                downstream=None,# downstream task, include classification and regression
                resizeshape=80,
                 ):
        super().__init__()
        self.resizeshape = resizeshape
        self.downstream = downstream
        self.scaler = scaler
        self.transform=transform
        self.augment=augment
        self.args=args
        self.data = pickle.load(open(data_path, 'rb'))
        
        prefix = data_path.split('/')[-1].split('_')[0]
        if prefix == 'trainval':
            prefix = 'train'
            
        # if prefix == 'train':
        #     self.healthy_data = pickle.load(open('/mnt/data2/ECG_CMR/UKB_select_data/train_data_v11_healthy.pt', 'rb'))
        # else:
        #     self.healthy_data = pickle.load(open('/mnt/data2/ECG_CMR/UKB_select_data/test_data_v11_healthy.pt', 'rb'))

        self.ecg = self.data[str(prefix+'_ecg_data')]
        self.select_tar = self.data[str(prefix+'_select_tar_data')]
        self.label = torch.from_numpy(self.data[str(prefix+'_label')])
        self.cmr = self.data[str(prefix+'_cmr_data')]
        self.la_cmr = torch.from_numpy(self.data[str(prefix+'_la_cmr_data')])
        self.eid = self.data[str(prefix+'_eid')]

        # self.health_ecg = self.healthy_data[str(prefix+'_ecg_data')][:len(self.ecg)]
        # self.health_select_tar = self.healthy_data[str(prefix+'_select_tar_data')][:len(self.ecg)]
        # self.health_label = torch.from_numpy(self.healthy_data[str(prefix+'_label')][:len(self.ecg)])
        # self.health_cmr = self.healthy_data[str(prefix+'_cmr_data')][:len(self.ecg)]
        # self.health_la_cmr = torch.from_numpy(self.healthy_data[str(prefix+'_la_cmr_data')][:len(self.ecg)])
        

        # self.label = torch.from_numpy(np.array([1]*len(self.ecg) + [0]*len(self.health_ecg)))
        # self.ecg = torch.cat((self.ecg, self.health_ecg), dim=0)
        # self.select_tar = torch.cat((self.select_tar, self.health_select_tar), dim=0)
        # self.cmr = torch.cat((self.cmr, self.health_cmr), dim=0)
        # self.la_cmr = torch.cat((self.la_cmr, self.health_la_cmr), dim=0)
        
        # del self.health_ecg, self.health_select_tar, self.health_label, self.health_cmr, self.health_la_cmr
        # del self.healthy_data
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
            if self.resizeshape is not None:
                cmr_transform = transforms.Compose([
                    transforms.Resize(size=self.resizeshape),
                    # transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
                ])
            # else:
            #     cmr_transform = transforms.Compose([
            #         transforms.Normalize(mean=[0.5]*50, std=[0.5]*50),
            #     ])
            cmr = cmr_transform(cmr).unsqueeze(1).repeat(1,3,1,1)
            la_cmr = cmr_transform(la_cmr).unsqueeze(1).repeat(1,3,1,1)

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
            cmr = cmr_augment(cmr).unsqueeze(1).repeat(1,3,1,1)
            la_cmr = cmr_augment(la_cmr).unsqueeze(1).repeat(1,3,1,1)
        batch = {
                'ecg': ecg,
                'select_tar': select_tar,
                'cha': self.label[index],
                'cmr': cmr,
                'la_cmr': la_cmr,
                'eid': self.eid[index]
            }
        return batch


@ECGCMRDATASET_zheyi.register_class()
class mutimodal_dataset_zheyi_ECG(data.Dataset):
    def __init__(self,data_path=None,
                transform=True,
                augment=False,
                args=None,
                scaler=None,
                resizeshape=None,
                downstream=None,# downstream task, include classification and regression
                 ):
        super().__init__()
        self.downstream = downstream
        self.transform=transform
        self.augment=augment
        self.args=args
        # self.train_data = pickle.load(open('/mnt/data2/ECG_CMR/zheyi_data/zheyi_train_data_unique.pkl', 'rb'))
        # self.test_data = pickle.load(open('/mnt/data2/ECG_CMR/zheyi_data/zheyi_test_data_unique.pkl','rb'))
        self.data = pickle.load(open(data_path, 'rb'))
        self.ecg = torch.from_numpy(np.array(self.data['ecg'])).float()
        self.eid = self.data['binglihao']
        self.xml_path = self.data['xml_path']
        # if type(self.train_data['ecg']) == list:
        #     self.data['ecg'] = np.array(self.train_data['ecg'] + self.test_data['ecg'])
        # else:
        #     self.data['ecg'] = np.concatenate((self.train_data['ecg'], self.test_data['ecg']), axis=0)

        # self.data['label'] = np.array(self.train_data['label'] + self.test_data['label'])
        # self.data['binglihao'] = self.train_data['binglihao'] + self.test_data['binglihao']
        # self.data['xml_path'] = self.train_data['xml_path'] + self.test_data['xml_path']

        # indices1 = np.where(self.data['label'] == 1)[0]
        # if len(indices1) > 500:
        #     indices1 = np.random.choice(indices1, 500, replace=False)
        # indices2 = np.where(self.data['label'] == 2)[0]
        # if len(indices2) > 500:
        #     indices2 = np.random.choice(indices2, 500, replace=False)
        # indices3 = np.where(self.data['label'] == 3)[0]
        # if len(indices3) > 500:
        #     indices3 = np.random.choice(indices3, 500, replace=False)
        
        # indices = np.concatenate((indices1, indices2, indices3), axis=0)
        # indices = np.where(self.data['label'] == 0)[0]
        # if len(indices) > 500:
        #     indices = np.random.choice(indices, 500, replace=False)

        # self.ecg = torch.from_numpy(self.data['ecg'][indices]).float()
        # self.label = self.data['label'][indices]
        # self.eid = [self.data['binglihao'][i] for i in indices]
        # self.xml_path = [self.data['xml_path'][i] for i in indices]
        del self.data
        # del self.test_data
        # del self.train_data
        
        # count_zeros = np.sum(self.label == 0)
        # count_ones = np.sum(self.label == 1)
        # count_twos = np.sum(self.label == 2)
        # count_threes = np.sum(self.label == 3)
        # print(f'Number of 0s: {count_zeros}, Number of 1s: {count_ones}, Number of 2s: {count_twos}, Number of 3s: {count_threes}')

        print(f'''ecg.shape: {self.ecg.shape},ecg.dtype:{self.ecg.dtype}''')


    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.ecg)

    def __getitem__(self, index) -> Any:
        """return the sample with given index"""
        ecg = self.ecg[index]
        eid = str(self.eid[index])
        ori_ecg = ecg
        
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
            print('No need for augmentation')
            exit()
        # print(f'ecg.dtype:{ecg.dtype},select_tar.dtype:{select_tar.dtype},I42.dtype:{I42.dtype}')
        batch = {
            'ecg': ecg,
            'eid': eid,
            'xml_path': self.xml_path[index],
            'ori_ecg': ori_ecg
        }
        return batch

if __name__ == '__main__':
    dataset = mutimodal_dataset_zheyi_ECG(data_path="/mnt/data2/ECG_CMR/test_data_dict_v8.pt")
    ecg,eid = dataset.__getitem__(0)
    print(ecg.shape)
    print(eid)