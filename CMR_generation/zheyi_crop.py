import nibabel as nib
import os
from scipy.ndimage import zoom
import numpy as np

from scipy.ndimage import label, find_objects, binary_dilation
from scipy.ndimage import generate_binary_structure
import pandas as pd

import torch

xls_path = '/mnt/sda1/liziyu/CMR_data/task9_merge_ECGCMR_diag_final.xlsx'
cmr_path = '/mnt/sda1/liziyu/CMR_data/laCMR_data_seg'
df = pd.read_excel(xls_path)

cmr_list = {'eid': [], 'img': [], 'age': [], 'label': [], 'bbox': []}

for index, row in df.iterrows():
    print(f'processing {index} ...')
    # 假设 mask 是你的二值 NumPy 矩阵
    folder_name = row['La_CMR_path'].split('.')[0]
    data_path = f'/mnt/sda1/liziyu/CMR_data/laCMR_data_seg/{folder_name}'
    img = nib.load(os.path.join(data_path, 'la_4ch.nii.gz')).get_fdata()
    mask = nib.load(os.path.join(data_path, 'seg4_la_4ch.nii.gz')).get_fdata()
    cmr_year = int(row['La_CMR_path'].split('_')[0])
    orig_bbox = ()
    try:
        bboxes = []
        for i in range(50):
            structure = generate_binary_structure(2, 1)
            dilated_mask = binary_dilation(mask[:, :, 0, i], structure)
            labeled_array, num_features = label(dilated_mask)  # 标记连通区域
            sizes = np.bincount(labeled_array.ravel())  # 计算每个连通区域的大小
            max_label = sizes[1:].argmax() + 1  # 找到最大连通区域的标签（跳过0）

            # 获取最大连通区域的切片
            slices = find_objects(labeled_array == max_label)

            # 提取边界框（bounding box）
            if len(slices) > 0:
                bbox = slices[0]
                bboxes.append(bbox)
            else:
                print("No connected components found.")
        min_coords = [float('inf')] * len(bboxes[0])  # 初始化为无穷大
        max_coords = [-float('inf')] * len(bboxes[0])  # 初始化为无穷小

        # 遍历每个边界框，更新最小和最大坐标
        for bbox in bboxes:
            for dim in range(len(bbox)):
                min_coords[dim] = min(min_coords[dim], bbox[dim].start)
                max_coords[dim] = max(max_coords[dim], bbox[dim].stop)

        # 创建最小外接框
        bbox = tuple(slice(min_coord, max_coord) for min_coord, max_coord in zip(min_coords, max_coords))

        # 计算当前边界框的高度和宽度
        height = bbox[0].stop - bbox[0].start
        width = bbox[1].stop - bbox[1].start
        # print(height, width)
        # 原始边界框信息也要保存
        orig_bbox = bbox
        # 检查并调整高度和宽度
        if height < 96:
            height_increase = 96 - height
            bbox = (slice(bbox[0].start - height_increase // 2, bbox[0].stop + (height_increase + 1) // 2), bbox[1])  # 中心对称调整

        if width < 96:
            width_increase = 96 - width
            bbox = (bbox[0], slice(bbox[1].start - width_increase // 2, bbox[1].stop + (width_increase + 1) // 2))  # 中心对称调整

        # img缩放到96*96

        extracted_region = img[:, :, 0, :][bbox]

        zoom_factors = (96 / extracted_region.shape[0], 96 / extracted_region.shape[1], 1)

        # 使用 zoom 进行缩放
        img_resized = zoom(extracted_region, zoom_factors, order=3)
        # print(img_resized.shape)
        cmr_list['eid'].append(row['病历号'])
        cmr_list['img'].append(img_resized)
        cmr_list['age'].append(int(row['年龄'][:2]) - 2024 + cmr_year)
        cmr_list['label'].append(row['分组'])
        cmr_list['bbox'].append(orig_bbox)
    except:
        cmr_list['eid'].append(row['病历号'])
        cmr_list['img'].append(None)
        cmr_list['age'].append(int(row['年龄'][:2]) - 2024 + cmr_year)
        cmr_list['label'].append(row['分组'])
        cmr_list['bbox'].append(None)
    # break
# print(cmr_list)
torch.save(cmr_list, '/mnt/sda1/liziyu/CMR_data/zheyi_data_v1.pt')