import torch
from utils.preprocess import get_img2
import SimpleITK as sitk
import numpy as np

trainval_data = torch.load("/mnt/data2/ECG_CMR/trainval_onlycmr_data_dict_v11_dn.pt")
mask_list = []
print(trainval_data.keys())
for eid in trainval_data['train_eid']:
    data, mask = get_img2(int(eid))
    mask_list.append(mask)

mask_list = np.array(mask_list)
trainval_data['train_mask_data'] = mask_list
torch.save(trainval_data, '/mnt/data2/ECG_CMR/trainval_onlycmr_data_dict_v11_dn_addmask.pt')


test_data = torch.load("/mnt/data2/ECG_CMR/test_onlycmr_data_dict_v11.pt")
mask_list = []
print(test_data.keys())
for eid in test_data['test_eid']:
    data, mask = get_img2(int(eid))
    mask_list.append(mask)

mask_list = np.array(mask_list)
test_data['test_mask_data'] = mask_list
torch.save(test_data, '/mnt/data2/ECG_CMR/test_onlycmr_data_dict_v11_addmask.pt')