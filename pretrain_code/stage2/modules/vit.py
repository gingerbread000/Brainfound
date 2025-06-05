# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/25 20:40
# @Author  : Liangdi.Ma
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, pretrained=False, checkpoint='imagenet', freeze_pretrained_layers=[], **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        if pretrained:
            self.load_pretrained_weights(checkpoint)

        if kwargs.get('freeze_pretraind_layers') and len(kwargs['freeze_pretraind_layers']) > 0:
            for layer_idx in kwargs['freeze_pretraind_layers']:
                for param in list(self.blocks[layer_idx].parameters()):
                    param.requires_grad = False

    def load_pretrained_weights(self, m):
        if m == 'mae':
            checkpoint = torch.load('/GPUFS/gyfyy_jxhe_1/User/maliangdi/models/vit/mae_pretrain_vit_base.pth',
                                    map_location='cpu')['model']
        elif m == 'imagenet':
            checkpoint = torch.load('/GPUFS/gyfyy_jxhe_1/User/maliangdi/models/vit/vit_base_patch16_224_in21k.pth',
                                    map_location='cpu')
        else:
            raise NotImplementedError
        patch_embed_dict = OrderedDict()
        blocks_dict = OrderedDict()
        norm_dict = OrderedDict()

        for k, v in checkpoint.items():
            if k.startswith('patch_embed'):
                patch_embed_dict[k[len('patch_embed.'):]] = v
            if k.startswith('blocks'):
                blocks_dict[k[len('blocks.'):]] = v
            if k.startswith('norm'):
                norm_dict[k[len('norm.'):]] = v
        self.patch_embed.load_state_dict(patch_embed_dict)
        self.blocks.load_state_dict(blocks_dict)
        self.norm.load_state_dict(norm_dict)
        print(f"Load {m} pretrained vision encoder checkpoint.")

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

if __name__ == '__main__':
    model = vit_base_patch16(global_pool=False)
    target_keys = list(model.state_dict().keys())
    source_keys = open('/GPUFS/gyfyy_jxhe_1/User/maliangdi/code/medication_pretrain/output/pretrain/Apr29_09-47_queue_size_slice=16384,queue_size_scan=2048,vision_decoder_depth=4,vison_decoder_heads=8,slice_pooler=mean,lr=1e-6,batch_size=2,gpu_num=8,seed=0/model_keys.txt').read()
    source_keys = [k[len('vision_encoder_q.'):] for k in source_keys.split(', ') if 'vision_encoder_q' in k]
    print(f"source({len(source_keys)}): ", source_keys)
    print("\n")
    print(f"target({len(target_keys)}): ", target_keys)
