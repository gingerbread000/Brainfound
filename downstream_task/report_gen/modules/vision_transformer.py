# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/06/25 16:40
# @Author  : Liangdi.Ma
from transformers import ViTFeatureExtractor, ViTModel
import torch
from torch import nn

class VisionTransformer(nn.Module):
    def __init__(self, config, pretrain=False, freeze=False, freeze_layer=None):
        super(VisionTransformer, self).__init__()
        if pretrain:
            self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
            self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

