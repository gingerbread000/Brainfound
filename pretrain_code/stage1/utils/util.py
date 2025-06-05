import os
import cv2
import json
import pickle
import natsort
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

from monai import transforms
from monai.transforms import apply_transform
import pandas as pd

def save_best_preds(mdict, path):
    with open(path, "wb") as f:
        pickle.dump(mdict, f)

class MDataset(Dataset):
    def __init__(self, section=True, transform=None, prompt_embeding=None) -> None:
        self.section = section
        self.transform = transform
        self.paths = []
        total_path_pkl = "{path to total file}.pkl"
        if os.path.exists(total_path_pkl):
            with open(total_path_pkl, "rb") as f:
                self.paths = pickle.load(f)

        self.ww_wc_list = [(160, 40), (80, 40), (1400, 400), (250, 40)]
        
        self.prompt_embeding = prompt_embeding

        print(len(self.paths)) 
    def __len__(self):
        return len(self.paths)

    def ww_wc(self, data, w_width=100, w_center=40):
        val_min = w_center - (w_width / 2)
        val_max = w_center + (w_width / 2)

        data_adjusted = data.copy()
        data_adjusted[data < val_min] = val_min
        data_adjusted[data > val_max] = val_max
        data_adjusted = (data_adjusted - data_adjusted.min()) / (1e-8 + data_adjusted.max() - data_adjusted.min())
        return data_adjusted
    
    def ret_mask(self, data): 
        d, h, w = data.shape                                                                                                                
        out = []
        for i in range(d):
            contours, hierarchy = cv2.findContours( 
                (data[i]>0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                                                   
            mask = cv2.fillPoly(np.zeros_like(data)[i].astype(np.uint8), contours, 1)
            kernel = np.ones((11, 11), dtype=np.uint8)
            dilate = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 5)
            out.append(dilate)  
        return np.array(out)
    
    def read_npy(self, fp):
        try:
            data_i = np.load(fp)
            if "mri" in fp:
                img = (data_i - data_i.min()) / (data_i.max() - data_i.min()+1e-8).astype(np.float32)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = (self.ww_wc(data_i, *self.ww_wc_list[np.random.randint(4)]) * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) / 255.
            
            img = np.transpose(img, (2,0,1))
            
        except Exception as e:
            print(f"wrong file: {fp} and wrong info: {e}")
            fp = self.paths[0]
            data_i = np.load(fp)
            if "mri" in fp:
                img = (data_i - data_i.min()) / (data_i.max() - data_i.min()+1e-8).astype(np.float32)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = (self.ww_wc(data_i, *self.ww_wc_list[np.random.randint(4)]) * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) / 255.
            img = np.transpose(img, (2,0,1))
        return img

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
        ind = 0
        fp = self.paths[index]
        if "t1" in fp.lower():
            ind = 1
        elif "t2" in fp.lower():
            ind = 2
        elif "pub_mri" in fp.lower():
            ind = 3

        return self._transform(fp), self.prompt_embeding[ind], ind
    
def get_dl(config,prompt_embeding):
    train_transforms = transforms.Compose([
            transforms.RandScaleCrop([0.9,0.9],[1.1,1.1],random_size=True),
            transforms.Resize([config.image_size,config.image_size]),
            transforms.RandFlip(prob=0.5, spatial_axis=0),
            transforms.RandFlip(prob=0.5, spatial_axis=1),
            transforms.RandRotate90(prob=0.3),
            transforms.ToTensor(),
            # transforms.RandAdjustContrast(prob=0.1, gamma=(0.97, 1.03)),
            # transforms.ThresholdIntensity(threshold=1, above=False, cval=1.0),
            # transforms.ThresholdIntensity(threshold=0, above=True, cval=0),
            transforms.NormalizeIntensity(0.5, 0.5),
        ])

    dataset = MDataset(section="train",transform=train_transforms,prompt_embeding=prompt_embeding)
    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)

    return train_dataloader, dataset
