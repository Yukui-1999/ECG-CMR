import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import pickle
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from scipy.signal import butter, resample, sosfiltfilt, square
from typing import Any, Dict, List, Optional, Tuple, Union
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


class ECG_genCMR(Dataset):
    def __init__(self,data_path):
        data = pickle.load(open(data_path, 'rb'))
        ecg, labels, path = zip(*data)
        self.ecg = list(ecg)
        self.labels = np.array(list(labels))
        self.path = list(path)
        
        self.ecg_transforms = Compose([
            Resample(target_fs=250),
            RandomCrop(2250),
            HighpassFilter(250, 0.67),
            LowpassFilter(250, 40),
            Standardize(axis=(-1, -2)),
            ToTensor()
        ])
    
    def __len__(self):
        return len(self.ecg)


    def __getitem__(self, i):
        item = self.path[i]
        item = item.split('.')[0].replace(',','_')
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)


        return ecg, item




class ECG_zheergenCMR(Dataset):
    def __init__(self,data_path):
        data = pickle.load(open(data_path, 'rb'))
        self.ecg = data['ecg']
        self.labels = data['label']
        self.eid = [i[0].split('\\')[-1].split('.')[0] for i in data['path']]
        
        self.ecg_transforms = Compose([
            Resample(target_fs=250),
            RandomCrop(2250),
            HighpassFilter(250, 0.67),
            LowpassFilter(250, 40),
            Standardize(axis=(-1, -2)),
            ToTensor()
        ])
    
    def __len__(self):
        return len(self.ecg)


    def __getitem__(self, i):
        item = self.eid[i]
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)


        return ecg, item

class ECG_quzhougenCMR(Dataset):
    def __init__(self,data_path):
        data = pickle.load(open(data_path, 'rb'))
        self.ecg = data['ecg']
        self.labels = np.array(data['label'])
        self.eid = [i.split('\\')[-3] +'-' + i.split('\\')[-2] +'-' + i.split('\\')[-1].split('.')[0] for i in data['path']]
        
        self.ecg_transforms = Compose([
            Resample(target_fs=250),
            RandomCrop(2250),
            HighpassFilter(250, 0.67),
            LowpassFilter(250, 40),
            Standardize(axis=(-1, -2)),
            ToTensor()
        ])
    
    def __len__(self):
        return len(self.ecg)


    def __getitem__(self, i):
        item = self.eid[i]
        ecg = self.ecg[i]
        ecg = self.ecg_transforms(ecg)


        return ecg, item


import wfdb
class ECGBaseMIMICDis(Dataset):
    def __init__(self,data):
        
        self.ecg_path = json.load(open(data, 'r'))
        
        self.ecg_transforms = Compose([
            Resample(target_fs=250),
            RandomCrop(2250),
            HighpassFilter(250, 0.67),
            LowpassFilter(250, 40),
            Standardize(axis=(-1, -2)),
            ToTensor()
        ])
    def __len__(self):
        return len(self.ecg_path)
    
    def __getitem__(self, i):
        item = self.ecg_path[i]
        eid = item.split('/')[-1].split('.dat')[0]
        rd_record = wfdb.rdrecord(item.split('.dat')[0])
        ecg = rd_record.p_signal.T
        ecg = self.ecg_transforms(ecg)
       
        return ecg, eid


class ECGyingguo(Dataset):
    def __init__(self,txt_file):

        self.ecg_path = '/mnt/sda1/liziyu/CMR_data/yingguo/processed_ecg'
        self.data = txt_file
        if '/' in self.data[0]:
            self.eid = [i[0].split('/')[-1].split('_')[0] for i in self.data]
        else:
            self.eid = self.data
        # print(self.data)
        # print(self.eid)


        self.ecg_transforms = Compose([
            Resample(target_fs=250),
            RandomCrop(2250),
            HighpassFilter(250, 0.67),
            LowpassFilter(250, 40),
            Standardize(axis=(-1, -2)),
            ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        eid = self.eid[i]
        
        ecg = pkl.load(open(os.path.join(self.ecg_path, eid + '__20205_2_0.pkl'), 'rb'))
        ecg = self.ecg_transforms(ecg)
        return ecg, eid



class simple_generate(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True) # (samples, 12, length)
        self.ecg_transforms = Compose([
            Resample(target_fs=250),
            RandomCrop(2250),
            HighpassFilter(250, 0.67),
            LowpassFilter(250, 40),
            Standardize(axis=(-1, -2)),
            ToTensor()
        ])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ecg = self.data[i]
        ecg = self.ecg_transforms(ecg)
        return ecg, f'sample_{i:04d}'  # Return a sample name based on index