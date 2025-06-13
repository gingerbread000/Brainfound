# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/06/05 14:17
# @Author  : Liangdi.Ma
import torch
import torch.nn as nn
import torchvision
import collections

class VisionEncoder(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=False, freeze=False, freeze_layers=None):
        super(VisionEncoder, self).__init__()
        # # self.visual_feature_size = dim
        # self.cnn = getattr(torchvision.models, backbone)(pretrained=pretrained)
        # # Do nothing after the final residual stage.
        # self.in_feature = self.cnn.fc.in_features
        # self.cnn.fc = nn.Identity()

        model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.in_feature = model.fc.in_features
        modules = list(model.children())[:-2]
        self.visual_encoder = nn.Sequential(*modules)
        self.adptive_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        # Freeze all weights if specified.
        if freeze:
            if freeze_layers is None:
                for param in self.cnn.parameters():
                    param.requires_grad = False
                self.cnn.eval()
            else:
                for layer_idx in freeze_layers:
                    for param in list(self.cnn.layer[layer_idx].parameters()):
                        param.requires_grad = False

    def forward(self, x):
        feature_maps = self.visual_encoder(x)  # (bs, in_feature, h, w)
        batch_size, feat_size, _, _ = feature_maps.shape
        global_feature = self.adptive_avgpool(feature_maps).reshape(batch_size, -1)
        patch_feats = feature_maps.reshape(batch_size, feat_size, -1).permute(0, 2, 1)  # (bs, h*w, in_feature), in accordance with vit
        return global_feature, patch_feats

    @property
    def output_hidden_size(self):
        return self.in_feature
