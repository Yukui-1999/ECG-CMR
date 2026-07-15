import copy
import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, resample, sosfiltfilt, square
from typing import Any, Dict, List, Optional, Tuple, Union
import wfdb 
import nibabel as nib

import matplotlib.pyplot as plt
import numpy as np

def plot_ecg_12_leads(ecg_data):
    """
    Plots a 12-lead ECG in a 2x6 grid format.
    
    Parameters:
    - ecg_data: A 2D numpy array of shape (12, length) representing the 12 leads of ECG signals.
    
    Returns:
    - None
    """
    # Check if the input data has the correct shape
    if ecg_data.shape[0] != 12:
        raise ValueError("Input ECG data must have 12 leads (shape should be (12, length))")
    
    # Set up the 2x6 grid for the 12 leads
    fig, axs = plt.subplots(2, 6, figsize=(18, 6), sharex=True)
    
    # Flatten the 2D array of axes for easy iteration
    axs = axs.flatten()
    
    # Define the lead names (if desired)
    lead_names = [
        'Lead I', 'Lead II', 'Lead III', 'aVR', 'aVL', 'aVF', 
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
    ]
    
    # Plot each lead on the 2x6 grid
    for i in range(12):
        axs[i].plot(ecg_data[i, :])  # Plot each lead
        axs[i].set_title(lead_names[i])  # Set the title for each subplot
        axs[i].grid(True)  # Add grid for better visibility

    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Show the plot
    plt.savefig(f'/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/test_fig/ecg_12_leads.png')

# Example usage:
# Assuming `ecg_data` is a numpy array of shape (12, length)
# ecg_data = np.random.randn(12, 1000)  # Replace with actual ECG data
# plot_ecg_12_leads(ecg_data)


def image_normalization(image, scale=1, mode="2D"):
    if isinstance(image, np.ndarray) and np.iscomplexobj(image):
        image = np.abs(image)
    low = image.min()
    high = image.max()
    im_ = (image - low) / (high - low)
    if scale is not None:
        im_ = im_ * scale
    return im_

class Resample:
    """Resample the input sequence.
    """
    def __init__(self,
                 target_length: Optional[int] = None,
                 target_fs: Optional[int] = None) -> None:
        self.target_length = target_length
        self.target_fs = target_fs

    def __call__(self, x: np.ndarray, fs: Optional[int] = 500) -> np.ndarray:
        if fs and self.target_fs and fs != self.target_fs:
            x = resample(x, int(x.shape[1] * self.target_fs / fs), axis=1)
        elif self.target_length and x.shape[1] != self.target_length:
            x = resample(x, self.target_length, axis=1)
        return x



class Resample1000:
    """Resample the input sequence.
    """
    def __init__(self,
                 target_length: Optional[int] = None,
                 target_fs: Optional[int] = None) -> None:
        self.target_length = target_length
        self.target_fs = target_fs

    def __call__(self, x: np.ndarray, fs: Optional[int] = 1000) -> np.ndarray:
        if fs and self.target_fs and fs != self.target_fs:
            x = resample(x, int(x.shape[1] * self.target_fs / fs), axis=1)
        elif self.target_length and x.shape[1] != self.target_length:
            x = resample(x, self.target_length, axis=1)
        return x

class SOSFilter:
    """Apply SOS filter to the input sequence.
    """
    def __init__(self,
                 fs: int,
                 cutoff: float,
                 order: int = 5,
                 btype: str = 'highpass') -> None:
        self.sos = butter(order, cutoff, btype=btype, fs=fs, output='sos')

    def __call__(self, x):
        return sosfiltfilt(self.sos, x)

class RandomCrop:
    """Crop randomly the input sequence.
    """
    def __init__(self, crop_length: int) -> None:
        self.crop_length = crop_length

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
        start_idx = np.random.randint(0, x.shape[1] - self.crop_length + 1)
        return x[:, start_idx:start_idx + self.crop_length]

class NCrop:
    """Crop the input sequence to N segments with equally spaced intervals.
    """
    def __init__(self, crop_length: int, num_segments: int) -> None:
        self.crop_length = crop_length
        self.num_segments = num_segments

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
        start_idx = np.arange(start=0,
                              stop=x.shape[1] - self.crop_length + 1,
                              step=(x.shape[1] - self.crop_length) // (self.num_segments - 1))
        return np.stack([x[:, i:i + self.crop_length] for i in start_idx], axis=0)
    
class HighpassFilter(SOSFilter):
    """Apply highpass filter to the input sequence.
    """
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(HighpassFilter, self).__init__(fs, cutoff, order, btype='highpass')

class LowpassFilter(SOSFilter):
    """Apply lowpass filter to the input sequence.
    """
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(LowpassFilter, self).__init__(fs, cutoff, order, btype='lowpass')

class Standardize:
    """Standardize the input sequence.
    """
    def __init__(self, axis: Union[int, Tuple[int, ...], List[int]] = (-1, -2)) -> None:
        if isinstance(axis, list):
            axis = tuple(axis)
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        loc = np.mean(x, axis=self.axis, keepdims=True)
        scale = np.std(x, axis=self.axis, keepdims=True)
        # Set rst = 0 if std = 0
        return np.divide(x - loc, scale, out=np.zeros_like(x), where=scale != 0)
class Compose:
    """Compose several transforms together.
    """
    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            x = transform(x)
        return x

class ToTensor:
    """Convert ndarrays in sample to Tensors.
    """
    _DTYPES = {
        "float": torch.float32,
        "double": torch.float64,
        "int": torch.int32,
        "long": torch.int64,
    }

    def __init__(self, dtype: Union[str, torch.dtype] = torch.float32) -> None:
        if isinstance(dtype, str):
            assert dtype in self._DTYPES, f"Invalid dtype: {dtype}"
            dtype = self._DTYPES[dtype]
        self.dtype = dtype

    def __call__(self, x: Any) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype)


import pickle as pkl
import pickle
class Example_ECGBaseMIMICDis(Dataset):
    def __init__(self,data_path):
        self.data = pickle.load(open(data_path, 'rb'))
        self.ecg = self.data['ecg']
        self.label = self.data['label']
        self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        
    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        if np.isnan(ecg).any():
            print(f'Found NaN in ECG data at index {i}, replacing with zeros')
            ecg = np.nan_to_num(ecg)
        label = self.label[i]
        ecg = self.ecg_transforms(ecg)
       
        return ecg, label
class ECGCMRBase(Dataset):
    def __init__(self,txt_file,isTrain=True):
        print(f'Loading ECGCMR dataset from {txt_file}')
        self.json_file = txt_file
        self.ecg_path = '/mnt/sda1/liziyu/CMR_data/yingguo/processed_ecg'
        self.data = json.load(open(txt_file, "r"))
        self.eid = copy.deepcopy(self.data)
        for i in range(len(self.data)):
            file_name = self.data[i] + '_20208_2_0'
            self.data[i] = os.path.join(f'/home/liziyu/CMRGEN/cmrmar/cmr_data/yingguo/{file_name}', file_name+'.pt')

        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = torch.load(item)
        image= np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)
        # print(image.shape)  # (bs,) c, f, h, w

        ecg = pkl.load(open(os.path.join(self.ecg_path, item.split('/')[-1].split('_')[0] + '__20205_2_0.pkl'), 'rb'))
        ecg = self.ecg_transforms(ecg)
        return ecg, image

class ECGCMRTrain(ECGCMRBase):
    def __init__(self):
        super().__init__(txt_file="/home/liziyu/CMRGEN/ECGCMR/train_data_v1.json", isTrain=True )
class ECGCMRValidation(ECGCMRBase):
    def __init__(self):
        super().__init__(txt_file="/home/liziyu/CMRGEN/ECGCMR/test_data_v1.json", isTrain=False )





class ECGBasePhen(Dataset):
    def __init__(self,txt_file,isTrain=True):

        self.ecg_path = '/mnt/sda1/liziyu/CMR_data/yingguo/processed_ecg'
        self.data = txt_file
        self.eid = [i[0].split('/')[-1].split('_')[0] for i in self.data]
        self.phenotype = np.array([i[1] for i in self.data])
        self.scaler = StandardScaler()
        self.phenotype = self.scaler.fit_transform(self.phenotype)

        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i][0]
        phen = self.phenotype[i]
        ecg = pkl.load(open(os.path.join(self.ecg_path, item.split('/')[-1].split('_')[0] + '__20205_2_0.pkl'), 'rb'))
        ecg = self.ecg_transforms(ecg)
        return ecg, phen


class ECGPhen_wGenCMR(Dataset):
    def __init__(self,txt_file,isTrain=True):

        self.ecg_path = '/mnt/sda1/liziyu/CMR_data/yingguo/processed_ecg'
        self.data = txt_file
        self.eid = [i[0].split('/')[-1].split('_')[0] for i in self.data]
        self.phenotype = np.array([i[1] for i in self.data])
        self.scaler = StandardScaler()
        self.phenotype = self.scaler.fit_transform(self.phenotype)
        self.gen_cmr_path = '/mnt/sda1/dingzhengyao/Work/CMR_gen/yingguo/ecg_cmr/sample/'
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i][0]
        phen = self.phenotype[i]
        ecg = pkl.load(open(os.path.join(self.ecg_path, item.split('/')[-1].split('_')[0] + '__20205_2_0.pkl'), 'rb'))
        ecg = self.ecg_transforms(ecg)
        gen_cmr = nib.load(os.path.join(self.gen_cmr_path, self.eid[i] + '.nii')).get_fdata()
        samples = {
            'ecg': ecg,
            'gen_cmr': gen_cmr
        }
        return samples, phen


class ECGCMRPhen(Dataset):
    def __init__(self,txt_file,isTrain=True):

        self.ecg_path = '/mnt/sda1/liziyu/CMR_data/yingguo/processed_ecg'
        self.data = json.load(open(txt_file, "r"))
        self.eid = [i[0].split('/')[-1].split('_')[0] for i in self.data]
        
        self.phenotype = np.array([i[1] for i in self.data])
        if isTrain:
            print('Fitting scaler on training data')
            self.scaler = StandardScaler()
            self.phenotype = self.scaler.fit_transform(self.phenotype)
            pkl.dump(self.scaler, open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/scaler.pkl', 'wb'))
        else:
            print('Loading scaler from /home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/scaler.pkl')
            self.scaler = pkl.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/scaler.pkl', 'rb'))
            self.phenotype = self.scaler.transform(self.phenotype)
        
        self.cmr_path = '/home/liziyu/CMRGEN/cmrmar/cmr_data/yingguo/'
        
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i][0]
        phen = self.phenotype[i]
        ecg = pkl.load(open(os.path.join(self.ecg_path, item.split('/')[-1].split('_')[0] + '__20205_2_0.pkl'), 'rb'))
        ecg = self.ecg_transforms(ecg)
        image = torch.load(os.path.join(self.cmr_path, self.eid[i] + '_20208_2_0', self.eid[i] + '_20208_2_0.pt'))
        image= np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)
        return ecg, image, phen




class ECGCMR_cmrPCAfeature(Dataset):
    def __init__(self,txt_file,isTrain=True):

        self.ecg_path = '/mnt/sda1/liziyu/CMR_data/yingguo/processed_ecg'
        self.data = json.load(open(txt_file, "r"))
        self.eid = copy.deepcopy(self.data)
        self.cmr_pca_path = '/mnt/sda1/dingzhengyao/Work/ECG_CMR_CardioNets_v1/CMR_cls_features/PCA-128/'
        
        
        self.cmr_path = '/home/liziyu/CMRGEN/cmrmar/cmr_data/yingguo/'
        
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
 
        cmr_pca_feature = np.load(os.path.join(self.cmr_pca_path, self.eid[i] + '_cmr_cls_feature.npy'))
        ecg = pkl.load(open(os.path.join(self.ecg_path, self.eid[i] + '__20205_2_0.pkl'), 'rb'))
        ecg = self.ecg_transforms(ecg)
        image = torch.load(os.path.join(self.cmr_path, self.eid[i] + '_20208_2_0', self.eid[i] + '_20208_2_0.pt'))
        image= np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)
        return ecg, image, cmr_pca_feature






class ECGPhen_wGenCMR_debug(Dataset):
    def __init__(self,txt_file,isTrain=True):

        self.ecg_path = '/mnt/sda1/liziyu/CMR_data/yingguo/processed_ecg'
        self.data = txt_file
        self.eid = [i[0].split('/')[-1].split('_')[0] for i in self.data]
        self.phenotype = np.array([i[1] for i in self.data])
        self.scaler = StandardScaler()
        self.phenotype = self.scaler.fit_transform(self.phenotype)
        self.gen_cmr_path = '/mnt/sda1/dingzhengyao/Work/CMR_gen/yingguo/ecg_cmr/sample/'
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i][0]
        phen = self.phenotype[i]
        ecg = pkl.load(open(os.path.join(self.ecg_path, item.split('/')[-1].split('_')[0] + '__20205_2_0.pkl'), 'rb'))
        ecg = self.ecg_transforms(ecg)
        gen_cmr = nib.load(os.path.join(self.gen_cmr_path, self.eid[i] + '.nii')).get_fdata()
        samples = {
            'ecg': ecg,
            'gen_cmr': gen_cmr
        }
        return samples, phen

class CMRBasePhen(Dataset):
    def __init__(self,txt_file,isTrain=True):
        self.data = txt_file
        self.eid = [i[0].split('/')[-1].split('_')[0] for i in self.data]
        self.phenotype = np.array([i[1] for i in self.data])
        if isTrain:
            self.scaler = StandardScaler()
            self.phenotype = self.scaler.fit_transform(self.phenotype)
            pkl.dump(self.scaler, open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/temp_scaler.pkl', 'wb'))
        else:
            self.scaler = pkl.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/temp_scaler.pkl', 'rb'))
            self.phenotype = self.scaler.transform(self.phenotype)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        cmr_path = self.data[index][0]
        metrics = self.phenotype[index]
        
        data = torch.load(os.path.join(cmr_path))
        img = np.transpose(data['img'], (2, 0, 1))
        img = (image_normalization(img)-0.5)*2
        img = torch.from_numpy(img)
        img = img.float()

        return img, metrics



class CMRBasePhen_gen(Dataset):
    def __init__(self,txt_file,isTrain=True):
        self.data = txt_file
        self.eid = [i[0].split('/')[-1].split('_')[0] for i in self.data]
        self.cmr_path_dir = '/mnt/sda1/dingzhengyao/Work/CMR_gen/yingguo/nejm_v1_fs8/sample/'
        self.phenotype = np.array([i[1] for i in self.data])
        self.scaler = StandardScaler()
        self.phenotype = self.scaler.fit_transform(self.phenotype)
        if isTrain:
            self.scaler = StandardScaler()
            self.phenotype = self.scaler.fit_transform(self.phenotype)
            pkl.dump(self.scaler, open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/temp_scaler.pkl', 'wb'))
        else:
            self.scaler = pkl.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/temp_scaler.pkl', 'rb'))
            self.phenotype = self.scaler.transform(self.phenotype)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        eid = self.eid[index]
        metrics = self.phenotype[index]
        
        img_nii = nib.load(os.path.join(self.cmr_path_dir, str(eid)+'.nii'))
        img = img_nii.get_fdata()
        img = (image_normalization(img)-0.5)*2
        img = torch.from_numpy(img)
        img = img.float()

        return img, metrics
    




class HeNan_ECGBaseDis(Dataset):
    def __init__(self,data,isTrain=True):


        self.ecg_path = '/mnt/sda1/dingzhengyao/Work/ECG_CMR_HenanDATA/ECG'
        self.data = data
        eid, labels = zip(*self.data)
        self.eid = list(eid)
        self.labels = list(labels)
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')

        if isTrain:
            self.ecg_transforms = Compose([
                Resample1000(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample1000(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])

    def __len__(self):
        return len(self.eid)

    def __getitem__(self, i):
        item = self.eid[i]
        label = self.labels[i]
        ecg = pkl.load(open(os.path.join(self.ecg_path, item + '.pkl'), 'rb'))
        ecg = self.ecg_transforms(ecg)
        return ecg, label









class ECGBaseDis(Dataset):
    def __init__(self,data,isTrain=True):


        self.ecg_path = '/mnt/sda1/liziyu/CMR_data/yingguo/processed_ecg'
        self.data = data
        eid, labels = zip(*self.data)
        self.eid = list(eid)
        self.labels = list(labels)
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')

        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])

    def __len__(self):
        return len(self.eid)

    def __getitem__(self, i):
        item = self.eid[i]
        label = self.labels[i]
        ecg = pkl.load(open(os.path.join(self.ecg_path, item + '__20205_2_0.pkl'), 'rb'))
        ecg = self.ecg_transforms(ecg)
        return ecg, label




class ECGDis_wGenCMR(Dataset):
    def __init__(self,data,isTrain=True):


        self.ecg_path = '/mnt/sda1/liziyu/CMR_data/yingguo/processed_ecg'
        self.data = data
        eid, labels = zip(*self.data)
        self.eid = list(eid)
        self.labels = list(labels)
        self.gen_cmr_path = '/mnt/sda1/dingzhengyao/Work/CMR_gen/yingguo/ecg_cmr/sample/'
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')

        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])

    def __len__(self):
        return len(self.eid)

    def __getitem__(self, i):
        item = self.eid[i]
        label = self.labels[i]
        ecg = pkl.load(open(os.path.join(self.ecg_path, item + '__20205_2_0.pkl'), 'rb'))
        ecg = self.ecg_transforms(ecg)
        gen_cmr = nib.load(os.path.join(self.gen_cmr_path, self.eid[i] + '.nii')).get_fdata()
        samples = {
            'ecg': ecg,
            'gen_cmr': gen_cmr
        }
        return samples, label





class CMRBaseDis(Dataset):
    def __init__(self,data):
        self.data = data
        eid, labels = zip(*self.data)
        self.eid = list(eid)
        self.labels = list(labels)
        self.cmr_path = '/home/liziyu/CMRGEN/cmrmar/cmr_data/yingguo'
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        
        label = self.labels[i]
        image = torch.load(os.path.join(self.cmr_path, self.eid[i] + '_20208_2_0', self.eid[i] + '_20208_2_0.pt'))
        image= np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)

        return image, label







class ECGBaseMIMICDis(Dataset):
    def __init__(self,data,train_percent=1,isTrain=True):
        self.data = data
        ecg_path, labels = zip(*self.data)
        self.ecg_path = list(ecg_path)
        self.labels = list(labels)
        self.low_quality_ecg_path = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_align/MIMIC_process/mimic_json/low_quality_mimic_data_path.json', 'r'))
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        if train_percent is not None:
            print(f'Using {train_percent} of the data for training')
            self.ecg_path = self.ecg_path[:int(len(self.ecg_path) * train_percent)]
            self.labels = self.labels[:int(len(self.labels) * train_percent)]
        # Remove low quality data
        self.good_quality_ecg_path = [path for path in self.ecg_path if path not in self.low_quality_ecg_path]
        self.good_quality_labels = [self.labels[i] for i in range(len(self.labels)) if self.ecg_path[i] not in self.low_quality_ecg_path]
        
        assert len(self.good_quality_ecg_path) == len(self.good_quality_labels), "Mismatch between ECG paths and labels after filtering"
        print(f'Using {len(self.good_quality_ecg_path)} samples after removing low quality data')
        # 统计一下正负样本的数量
        pos_count = sum(label == 1 for label in self.good_quality_labels)
        neg_count = sum(label == 0 for label in self.good_quality_labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    
    def __len__(self):
        return len(self.good_quality_ecg_path)
    
    def __getitem__(self, i):
        item = self.good_quality_ecg_path[i]
        label = self.good_quality_labels[i]
        rd_record = wfdb.rdrecord(item.split('.dat')[0])
        ecg = rd_record.p_signal.T
        ecg = self.ecg_transforms(ecg)
       
        return ecg, label
        





class ECGMIMICDis_wGenCMR(Dataset):
    def __init__(self,data,train_percent=1,isTrain=True):
        self.data = data
        ecg_path, labels = zip(*self.data)
        self.ecg_path = list(ecg_path)
        self.labels = list(labels)
        self.low_quality_ecg_path = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_align/MIMIC_process/mimic_json/low_quality_mimic_data_path.json', 'r'))
        self.gen_cmr_path = '/mnt/sda1/dingzhengyao/Work/CMR_gen/mimic/ecg_cmr/sample/'
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        if train_percent is not None:
            print(f'Using {train_percent} of the data for training')
            self.ecg_path = self.ecg_path[:int(len(self.ecg_path) * train_percent)]
            self.labels = self.labels[:int(len(self.labels) * train_percent)]
        # Remove low quality data
        self.good_quality_ecg_path = [path for path in self.ecg_path if path not in self.low_quality_ecg_path]
        self.good_quality_labels = [self.labels[i] for i in range(len(self.labels)) if self.ecg_path[i] not in self.low_quality_ecg_path]
        self.good_eid = [i.split('/')[-1].split('.dat')[0] for i in self.good_quality_ecg_path]
        assert len(self.good_quality_ecg_path) == len(self.good_quality_labels), "Mismatch between ECG paths and labels after filtering"
        print(f'Using {len(self.good_quality_ecg_path)} samples after removing low quality data')
        # 统计一下正负样本的数量
        pos_count = sum(label == 1 for label in self.good_quality_labels)
        neg_count = sum(label == 0 for label in self.good_quality_labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    
    def __len__(self):
        return len(self.good_quality_ecg_path)
    
    def __getitem__(self, i):
        item = self.good_quality_ecg_path[i]
        label = self.good_quality_labels[i]
        rd_record = wfdb.rdrecord(item.split('.dat')[0])
        ecg = rd_record.p_signal.T
        ecg = self.ecg_transforms(ecg)
        gen_cmr = nib.load(os.path.join(self.gen_cmr_path, self.good_eid[i] + '.nii')).get_fdata()
        samples = {
            'ecg': ecg,
            'gen_cmr': gen_cmr
        }
        return samples, label
        





class ECGBaseMIMIC_CMthree(Dataset):
    def __init__(self,data,isTrain=True):
        self.data = data
        ecg_path, labels = zip(*self.data)
        self.ecg_path = list(ecg_path)
        self.labels = np.array(list(labels))
        self.low_quality_ecg_path = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_align/MIMIC_process/mimic_json/low_quality_mimic_CMT_data_path.json', 'r'))
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        
        # Remove low quality data
        self.good_quality_ecg_path = [path for path in self.ecg_path if path not in self.low_quality_ecg_path]
        self.good_quality_labels = [self.labels[i] for i in range(len(self.labels)) if self.ecg_path[i] not in self.low_quality_ecg_path]
        
        assert len(self.good_quality_ecg_path) == len(self.good_quality_labels), "Mismatch between ECG paths and labels after filtering"
        print(f'Using {len(self.good_quality_ecg_path)} samples after removing low quality data')
        
        count_ones = np.sum(self.labels == 1)
        count_twos = np.sum(self.labels == 2)
        count_threes = np.sum(self.labels == 0)
        print(f'Number of 1s DCM: {count_ones}, Number of 2s HCM: {count_twos}, Number of 0s RCM: {count_threes}')
    
    def __len__(self):
        return len(self.good_quality_ecg_path)
    
    def __getitem__(self, i):
        item = self.good_quality_ecg_path[i]
        label = self.good_quality_labels[i]
        rd_record = wfdb.rdrecord(item.split('.dat')[0])
        ecg = rd_record.p_signal.T
        ecg = self.ecg_transforms(ecg)
       
        return ecg, label


class ECGMIMIC_CMthree_wGenCMR(Dataset):
    def __init__(self,data,isTrain=True):
        self.data = data
        ecg_path, labels = zip(*self.data)
        self.ecg_path = list(ecg_path)
        self.labels = np.array(list(labels))
        self.low_quality_ecg_path = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_align/MIMIC_process/mimic_json/low_quality_mimic_CMT_data_path.json', 'r'))
        self.gen_cmr_path = '/mnt/sda1/dingzhengyao/Work/CMR_gen/mimic/ecg_cmr/sample/'
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        
        # Remove low quality data
        self.good_quality_ecg_path = [path for path in self.ecg_path if path not in self.low_quality_ecg_path]
        self.good_quality_labels = [self.labels[i] for i in range(len(self.labels)) if self.ecg_path[i] not in self.low_quality_ecg_path]
        self.good_eid = [i.split('/')[-1].split('.dat')[0] for i in self.good_quality_ecg_path]
        assert len(self.good_quality_ecg_path) == len(self.good_quality_labels), "Mismatch between ECG paths and labels after filtering"
        print(f'Using {len(self.good_quality_ecg_path)} samples after removing low quality data')
        
        count_ones = np.sum(self.labels == 1)
        count_twos = np.sum(self.labels == 2)
        count_threes = np.sum(self.labels == 0)
        print(f'Number of 1s DCM: {count_ones}, Number of 2s HCM: {count_twos}, Number of 0s RCM: {count_threes}')
    
    def __len__(self):
        return len(self.good_quality_ecg_path)
    
    def __getitem__(self, i):
        item = self.good_quality_ecg_path[i]
        label = self.good_quality_labels[i]
        rd_record = wfdb.rdrecord(item.split('.dat')[0])
        ecg = rd_record.p_signal.T
        ecg = self.ecg_transforms(ecg)
        gen_cmr = nib.load(os.path.join(self.gen_cmr_path, self.good_eid[i] + '.nii')).get_fdata()
        samples = {
            'ecg': ecg,
            'gen_cmr': gen_cmr
        }
       
        return samples, label
    
        

class ECGzheyi_three_Base(Dataset):
    def __init__(self, data=None,isTrain=True):
        
        ecg, labels, path = zip(*data)
        self.ecg = list(ecg)
        self.labels = np.array(list(labels))
        self.path = list(path)
        print(self.labels)
        
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        
        count_ones = np.sum(self.labels == 1)
        count_twos = np.sum(self.labels == 2)
        count_threes = np.sum(self.labels == 0)
        print(f'Number of 1s DCM: {count_ones}, Number of 2s HCM: {count_twos}, Number of 0s RCM: {count_threes}')

    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)
        label = self.labels[i]
        return ecg, label


class ECGzheyi_three_WGenCMR(Dataset):
    def __init__(self, data=None,isTrain=True):
        
        ecg, labels, path = zip(*data)
        self.ecg = list(ecg)
        self.labels = np.array(list(labels))
        self.path = list(path)
        print(self.labels)
        self.gen_cmr_data = '/mnt/sda1/dingzhengyao/Work/CMR_gen/zheyi/ecg_cmr/sample/'
        
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        
        count_ones = np.sum(self.labels == 1)
        count_twos = np.sum(self.labels == 2)
        count_threes = np.sum(self.labels == 0)
        print(f'Number of 1s DCM: {count_ones}, Number of 2s HCM: {count_twos}, Number of 0s RCM: {count_threes}')

    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)
        label = self.labels[i]
        path = self.path[i].split('.')[0].replace(',','_')
        gen_cmr = nib.load(os.path.join(self.gen_cmr_data, path + '.nii')).get_fdata()
        samples = {
            'ecg': ecg,
            'gen_cmr': gen_cmr
        }
        return samples, label
    

# /home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/nejm_revision_zheyi_cmr/renjishiyan_addCMR.pkl 
# /mnt/data2/ECG_CMR/zheyi_data/Final_data/Fianl_ECGCMR_v2.pkl
class ECGzheyi_two_Base(Dataset):
    def __init__(self, data_path="/mnt/data2/ECG_CMR/zheyi_data/Final_data/Fianl_ECGCMR_v2.pkl",isTrain=True):
        data = pkl.load(open(data_path, "rb"))
        self.ecg = data['ecg']
        self.labels = data['label']
        self.labels = [ 0 if i == 0 else 1 for i in self.labels]
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        # 统计一下正负样本的数量
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    
    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)
        label = self.labels[i]
        return ecg, label


class ECGzheyi_two_BaseCMR(Dataset):
    def __init__(self, data_path="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/nejm_revision_zheyi_cmr/zheyi_data_final3.pkl",isTrain=True,args=None):
        if type(data_path) == str:
            data = pkl.load(open(data_path, "rb"))
        else:
            data = data_path
        self.args = args
        self.ecg = data['ecg']
        self.labels = data['label']
        self.labels = [ 0 if i == 0 else 1 for i in self.labels]
        self.cmr = data['img']
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        # 统计一下正负样本的数量
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    
    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)
        label = self.labels[i]
        img = self.cmr[i]
        img = np.transpose(img, (2, 0, 1))
        img = (image_normalization(img)-0.5)*2
        img = torch.from_numpy(img)
        img = img.float()
        if self.args.input_modality == 'ECG':
            return ecg, label
        elif self.args.input_modality == 'CMR':
            return img, label
        



class ECGzheyi_two_wGenCMR(Dataset):
    def __init__(self, data_path="/mnt/data2/ECG_CMR/zheyi_data/Final_data/Fianl_ECGCMR_v2.pkl",isTrain=True):
        data = pkl.load(open(data_path, "rb"))
        self.ecg = data['ecg']
        self.labels = data['label']
        self.labels = [ 0 if i == 0 else 1 for i in self.labels]
        self.eid = data['binglihao']
        self.gen_cmr_data = '/mnt/sda1/dingzhengyao/Work/CMR_gen/zheyi/ecg_cmr/sample/'
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        # 统计一下正负样本的数量
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    
    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)
        label = self.labels[i]
        gen_cmr = nib.load(os.path.join(self.gen_cmr_data, self.eid[i] + '.nii')).get_fdata()
        samples = {
            'ecg': ecg,
            'gen_cmr': gen_cmr
        }
        return samples, label

    
    

class ECGzheer_two_Base(Dataset):
    def __init__(self, data_path="/mnt/data2/ECG_CMR/zheer_data/finetuned_data/zheer_new_data.pkl",isTrain=True):
        if type(data_path) == str:
            data = pkl.load(open(data_path, "rb"))
        else:
            data = data_path
        self.ecg = data['ecg']
        self.labels = data['label']
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        # 统计一下正负样本的数量
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    
    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)
        label = self.labels[i]
        return ecg, label




class ECGzheer_two_wGenCMR(Dataset):
    def __init__(self, data_path="/mnt/data2/ECG_CMR/zheer_data/finetuned_data/zheer_new_data.pkl",isTrain=True):
        data = pkl.load(open(data_path, "rb"))
        self.ecg = data['ecg']
        self.labels = data['label']
        self.eid = [i[0].split('\\')[-1].split('.')[0] for i in data['path']]
        self.gen_cmr_data = '/mnt/sda1/dingzhengyao/Work/CMR_gen/zheer/ecg_cmr/sample/'
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        # 统计一下正负样本的数量
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    
    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)
        label = self.labels[i]
        gen_cmr = nib.load(os.path.join(self.gen_cmr_data, self.eid[i] + '.nii')).get_fdata()
        samples = {
            'ecg': ecg,
            'gen_cmr': gen_cmr
        }

        return samples, label





class ECGquzhou_two_Base(Dataset):
    def __init__(self, data_path="/mnt/data2/ECG_CMR/quzhou_data/quzhouECG_v3.pkl",isTrain=True):
        data = pkl.load(open(data_path, "rb"))
        self.ecg = data['ecg']
        self.labels = np.array(data['label'])
        self.labels[self.labels == 2] = 1
        print(self.labels)
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        # 统计一下正负样本的数量
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    
    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)
        label = self.labels[i]
        return ecg, label





class ECGquzhou_two_wGenCMR(Dataset):
    def __init__(self, data_path="/mnt/data2/ECG_CMR/quzhou_data/quzhouECG_v3.pkl",isTrain=True):
        data = pkl.load(open(data_path, "rb"))
        self.ecg = data['ecg']
        self.labels = np.array(data['label'])
        self.eid = [i.split('\\')[-3] +'-' + i.split('\\')[-2] +'-' + i.split('\\')[-1].split('.')[0] for i in data['path']]
        self.gen_cmr_data = '/mnt/sda1/dingzhengyao/Work/CMR_gen/quzhou/ecg_cmr/sample/'
        self.labels[self.labels == 2] = 1
        print(self.labels)
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        # 统计一下正负样本的数量
        pos_count = sum(label == 1 for label in self.labels)
        neg_count = sum(label == 0 for label in self.labels)
        print(f'Positive samples: {pos_count}, Negative samples: {neg_count}')
    
    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, i):
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)
        label = self.labels[i]
        gen_cmr = nib.load(os.path.join(self.gen_cmr_data, self.eid[i] + '.nii')).get_fdata()
        samples = {
            'ecg': ecg,
            'gen_cmr': gen_cmr
        }
        return samples, label


class renji_dataset(Dataset):
    def __init__(self, args,isTrain=False):
        data = pkl.load(open(args.data_path, "rb"))
        self.data_ecg = data['ecg']
        self.data_eid = data['binglihao']
        if isTrain:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                RandomCrop(2250),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
        else:
            self.ecg_transforms = Compose([
                Resample(target_fs=250),
                NCrop(2250, 3),
                HighpassFilter(250, 0.67),
                LowpassFilter(250, 40),
                Standardize(axis=(-1, -2)),
                ToTensor()
            ])
    
    def __len__(self):
        return len(self.data_ecg)
    def __getitem__(self, index):
        ecg = self.data_ecg[index]
        ecg = self.ecg_transforms(ecg)
        eid = self.data_eid[index]
        return ecg, eid