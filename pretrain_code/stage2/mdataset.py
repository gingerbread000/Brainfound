import os
import cv2
import json
import pickle
import natsort
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from collections import Counter

import pdb

import torch
from torch.utils.data import DataLoader, Dataset

from monai import transforms
from monai.transforms import apply_transform
from scipy.ndimage import zoom


def save_best_preds(mdict, path):
    with open(path, "wb") as f:
        pickle.dump(mdict, f)

def get_mod_prompt(tokenizer, mod_cls=0):
    raw_inputs = [
            "The format of input data is ct.",
            "The format of input data is t1 mri.",
            "The format of input data is t2 mri.",
        ]
    res = tokenizer(raw_inputs, padding="max_length", max_length=16, return_tensors="pt")
    text_embeding = res["input_ids"][mod_cls].view(1,-1,1)
    token_type_ids = res["token_type_ids"][mod_cls].view(1,-1,1)
    attention_mask = res["attention_mask"][mod_cls].view(1,-1,1)
    return text_embeding, attention_mask

class MDataset(Dataset):
    def __init__(self, configs, section="train") -> None:
        self.section = section
        if section == "train":
            self.transform = transforms.Compose([
                transforms.RandScaleCrop([1, 0.9,0.9],[1, 1.1,1.1],random_size=True),
                transforms.Resize([3, 256,256]),
                transforms.RandFlip(prob=0.5, spatial_axis=0),
                transforms.RandFlip(prob=0.5, spatial_axis=1),
                transforms.RandRotate(0.2,0.2),
                transforms.ToTensor(),
                transforms.RandAdjustContrast(prob=0.1, gamma=(0.97, 1.03)),
                transforms.ThresholdIntensity(threshold=1, above=False, cval=1.0),
                transforms.ThresholdIntensity(threshold=0, above=True, cval=0),
                transforms.NormalizeIntensity(0.5, 0.5),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([3, 256,256]),
                transforms.ToTensor(),
                transforms.ThresholdIntensity(threshold=1, above=False, cval=1.0),
                transforms.ThresholdIntensity(threshold=0, above=True, cval=0),
                transforms.NormalizeIntensity(0.5, 0.5),
            ])
        if section == "train":
            with open("/path_to_train_index/train.json", "r") as f:
                index_k = json.load(f)
        elif section == "valid":
            with open("/path_to_val_index/val.json", "r") as f:
                index_k = json.load(f)
        elif section == "test":
            with open("/path_to_test_index/test.json", "r") as f:
                index_k = json.load(f)

        with open("/path_infomation.json", "r") as f:
            info_dict = json.load(f)
        
        self.index_k = index_k
        self.k_list = list(index_k.keys())
        self.info_dict = info_dict

        # self.k_list = [k for k in self.k_list if self.only_one_cls(k)]
        print(len(self.k_list)) 

    def __len__(self):
        return len(self.k_list)
    
    def only_one_cls(self, k):
        return np.sum([self.info_dict[d][k] for d in ["is_normal", "is_hemorrhage", "is_infarction", "is_fracture", "is_tumor"]]) <=1

    def ww_wc(self, data, w_width=160, w_center=40):
        val_min = w_center - (w_width / 2)
        val_max = w_center + (w_width / 2)

        data_adjusted = data.copy()
        data_adjusted[data < val_min] = val_min
        data_adjusted[data > val_max] = val_max
        data_adjusted = (data_adjusted - data_adjusted.min()) / (1e-8 + data_adjusted.max() - data_adjusted.min())
        return data_adjusted

    def read_npy(self, k):
        try:
            fp = self.info_dict["new_path"][k]
            data_i = nib.load(fp).get_fdata()
        
            data_i = (self.ww_wc(data_i) * 255).astype(np.uint8)
            data_i = [cv2.cvtColor(data_i[:,:,i], cv2.COLOR_GRAY2RGB) / 255. for i in range(data_i.shape[2])]
            
        except Exception as e:
            print(f"wrong file: {k} and wrong info: {e}")
            fp = self.info_dict["new_path"]["0"]
            data_i = nib.load(fp).get_fdata()
        
            data_i = (self.ww_wc(data_i) * 255).astype(np.uint8)
            data_i = [cv2.cvtColor(data_i[:,:,i], cv2.COLOR_GRAY2RGB) / 255. for i in range(data_i.shape[2])]

        # print(data_i[0].shape, len(data_i))
        if len(data_i) > 32:
            data_i = data_i[1:33]
        else:
            pad_l = 32 - len(data_i)
            front = pad_l // 2
            behind = pad_l - front
            data_i = [np.zeros(data_i[0].shape[:])] * front + data_i + \
                [np.zeros(data_i[0].shape[:])] * behind
        
        data_i = np.array(data_i)
        data_i = np.transpose(data_i, (0, 3, 1, 2))
        # print(data_i.shape)
        return data_i

    def _transform(self, k):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.read_npy(k)
        return self.transform(data_i)
    
    def get_diag(self, k):
        disease_cls = 0
        diag = "疾病类别是："
        if self.info_dict["is_normal"][k] == 1:
            diag += "这是正常类别。"
            disease_cls = 0
        if self.info_dict["is_hemorrhage"][k] == 1:
            diag += "这是出血类别。"
            disease_cls = 1
        if self.info_dict["is_infarction"][k] == 1:
            diag += "这是缺血性类别。"
            disease_cls = 2
        if self.info_dict["is_fracture"][k] == 1:
            diag += "这是骨折类别。"
            disease_cls = 3
        if self.info_dict["is_tumor"][k] == 1:
            diag += "这是肿瘤类别。"
            disease_cls = 4
        if diag == "疾病类别是：":
            diag += "不是五类中的类别。"
            disease_cls = 5
        return diag, disease_cls

    def __getitem__(self, index):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        # pdb.set_trace()
        mod_cls = 0
        k = self.k_list[index]
        report = self.info_dict["DESCRIPTION"][k].strip().replace(" ", "")
        diag, disease_cls = self.get_diag(k)
        report = diag[6:]
        
        return {"image": [self._transform(k)], "caption":[report], "mod_cls": [mod_cls], "k": k, "cls":disease_cls} # text_embeding, attention_mask


class MDataset_EX(Dataset):
    def __init__(self, configs, section="train") -> None:
        self.section = section
        if section == "train":
            self.transform = transforms.Compose([
                transforms.RandScaleCrop([1, 0.9,0.9],[1, 1.1,1.1],random_size=True),
                transforms.Resize([3, 256,256]),
                transforms.RandFlip(prob=0.5, spatial_axis=0),
                transforms.RandFlip(prob=0.5, spatial_axis=1),
                transforms.RandRotate(0.2,0.2),
                transforms.ToTensor(),
                transforms.RandAdjustContrast(prob=0.1, gamma=(0.97, 1.03)),
                transforms.ThresholdIntensity(threshold=1, above=False, cval=1.0),
                transforms.ThresholdIntensity(threshold=0, above=True, cval=0),
                transforms.NormalizeIntensity(0.5, 0.5),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([3, 256,256]),
                transforms.ToTensor(),
                transforms.ThresholdIntensity(threshold=1, above=False, cval=1.0),
                transforms.ThresholdIntensity(threshold=0, above=True, cval=0),
                transforms.NormalizeIntensity(0.5, 0.5),
            ])

        self.paths = []
        
        label_map = {'正常':0, "出血":1, '脑梗':2, '骨折':3, '肿瘤':4}
        for root, dirs, files in os.walk("/path_to_external_data/"):
            for f in files:
                if f.endswith("nii.gz"):
                    self.paths.append([label_map[os.path.basename(root)], os.path.join(root, f)])
        print(len(self.paths)) 
        np.random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)
    
    def only_one_cls(self, k):
        return np.sum([self.info_dict[d][k] for d in ["is_normal", "is_hemorrhage", "is_infarction", "is_fracture", "is_tumor"]]) <=1

    def ww_wc(self, data, w_width=160, w_center=40):
        val_min = w_center - (w_width / 2)
        val_max = w_center + (w_width / 2)

        data_adjusted = data.copy()
        data_adjusted[data < val_min] = val_min
        data_adjusted[data > val_max] = val_max
        data_adjusted = (data_adjusted - data_adjusted.min()) / (1e-8 + data_adjusted.max() - data_adjusted.min())
        return data_adjusted

    def read_npy(self, fp):
        try:
            data_i = nib.load(fp).get_fdata()
        except Exception as e:
            print(f"wrong file: {fp} and wrong info: {e}")
            data_i = nib.load(fp).get_fdata()
        
        data_i = (self.ww_wc(data_i) * 255).astype(np.uint8)
        
        d, h, w = data_i.shape

        data_i = zoom(data_i, (32/data_i.shape[0], 1, 1)).astype(np.uint8)
        
        data_i = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) / 255. for img in data_i]

        data_i = np.array(data_i)
        data_i = np.transpose(data_i, (0, 3, 1, 2))
        
        return data_i

    def _transform(self, fp):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.read_npy(fp)
        return self.transform(data_i)

    def __getitem__(self, index):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        mod_cls = 0
        label, fp = self.paths[index]
        
        return {"image": [self._transform(fp)], "caption":["none"], "mod_cls": [mod_cls], "k": os.path.basename(fp), "cls":int(label)} # text_embeding, attention_mask


if __name__ == "__main__":
    import pdb

    d = MDataset(None)
    for i, item in enumerate(d):
        print(i)
        pdb.set_trace()
