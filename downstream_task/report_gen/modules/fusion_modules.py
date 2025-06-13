# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/07/14 17:24
# @Author  : Liangdi.Ma
import math
import torch
from torch import nn
from einops import repeat
import functools

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MeanPool(nn.Module):
    def __init__(self, in_features=2048):
        super(MeanPool, self).__init__()
        self.output_hidden_size = in_features
        self.avgpool1d = nn.AdaptiveAvgPool1d(output_size=1)
        self.avgpool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, seq_patch_feats):
        # patch_feats: (bs, seq_len, patch_num, dim)
        batch_size, seq_len, pnum, fdim = seq_patch_feats.shape
        pool_feature = self.avgpool2d(seq_patch_feats.permute(0, 3, 1, 2)).reshape(batch_size, -1)
        pool_embedding = self.avgpool1d(seq_patch_feats.permute(0, 2, 3, 1).reshape(-1, fdim, seq_len))
        pool_embedding = pool_embedding.reshape(batch_size, pnum, fdim)
        # pool_feature = (batch size, dim), pool_embedding = (batch size, patch_num, dim)
        return pool_feature, pool_embedding

class MultiDimFusionTransformer(nn.Module):
    def __init__(self, in_features=2048, hidden_size=768):
        super(MultiDimFusionTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.in_features = in_features
        self.output_hidden_size = hidden_size
        self.projection = nn.Linear(self.in_features, self.hidden_size)
        self.space_position = nn.Embedding(512, self.hidden_size)
        self.layer_position = nn.Embedding(512, self.hidden_size)
        self.space_cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.layer_cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.layer_norm = nn.LayerNorm(
            hidden_size, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=0.1)

        enc_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=8, dim_feedforward=1024)
        self.space_encoder = nn.TransformerEncoder(enc_layer, num_layers=12)
        self.layer_encoder = nn.TransformerEncoder(enc_layer, num_layers=12)

    def forward(self, seq_patch_feats):
        # patch_feats: (bs, seq_len, patch_num, dim)
        projected_feats = self.projection(seq_patch_feats)
        # print('projected feature shape: ', projected_feats.shape)
        batch_size, seq_len, pnum, fdim = projected_feats.shape
        projected_feats = projected_feats.reshape(-1, pnum, fdim)

        space_position_indices = torch.arange(pnum+1, dtype=torch.long, device=projected_feats.device)
        space_position_indices = space_position_indices.unsqueeze(0).expand(batch_size * seq_len, -1)
        space_position_embeddings = self.space_position(space_position_indices)
        # space_position_embeddings = space_position_embeddings.reshape(batch_size, seq_len, pnum+1, fdim)
        # shape: (batch_size, seq_len, patch_num, dim)
        # print('space_position_embeddings shape: ', space_position_embeddings.shape)

        space_cls_tokens = repeat(self.space_cls_token, '1 1 d -> b 1 d', b=batch_size*seq_len)
        space_input = torch.cat((space_cls_tokens, projected_feats), dim=1)
        space_input += space_position_embeddings
        space_input = self.layer_norm(space_input)
        space_input = self.dropout(space_input)  # (batch_size * seq_len, pnum, dim)
        # print(f'space input shape: {space_input.shape}')

        space_last_output_hidden = self.space_encoder(space_input.permute(1, 0, 2))  # (pnum + 1, batch_size * seq_len, dim)
        space_cls_hidden = space_last_output_hidden[0]  # (batch_size * seq_len, dim)
        # print('space cls hidden shape: ', space_cls_hidden.shape)
        space_cls_hidden = space_cls_hidden.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, dim)

        # layer position encoding
        layer_position_indices = torch.arange(seq_len + 1, dtype=torch.long, device=space_cls_hidden.device)
        layer_position_indices = layer_position_indices.unsqueeze(0).expand(batch_size, -1)
        layer_position_embeddings = self.layer_position(layer_position_indices)  # (batch_size, seq_len, dim)
        # print('layer_position_embeddings shape: ', layer_position_embeddings.shape)

        layer_cls_tokens = repeat(self.layer_cls_token, '1 1 d -> b 1 d', b=batch_size)
        layer_input = torch.cat((layer_cls_tokens, space_cls_hidden), dim=1)
        layer_input += layer_position_embeddings
        layer_input = self.layer_norm(layer_input)
        layer_input = self.dropout(layer_input)  # (batch_size, seq_len+1, dim)
        last_output_hidden = self.layer_encoder(layer_input.permute(1, 0, 2))  # (seq_len + 1, batch_size, dim)
        last_output_hidden = last_output_hidden.permute(1, 0, 2)  # (batch_size, seq_len + 1, dim)
        cls_output = last_output_hidden[:, 0]  # (batch_size, dim)
        output_embeddings = last_output_hidden[:, 1:]  # (batch_size, seq_len, dim)
        return cls_output, output_embeddings


class FusionTransformer(nn.Module):
    def __init__(self, embed_dim=768, depth=2, num_heads=8, dim_feedforward=3072, dropout=0.1):
        super(FusionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.pos_embed = nn.Embedding(512, self.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        try:
            enc_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, attn_drop=nn.Dropout(p=dropout))
        except:

            enc_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.layer_norm = nn.LayerNorm(
            embed_dim, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x: [batch size, seq num, d]
        batch_size, seq_len, fdim = x.shape
        # layer position encoding
        position_indices = torch.arange(seq_len + 1, dtype=torch.long, device=x.device)
        position_indices = position_indices.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.pos_embed(position_indices)  # (batch_size, seq_len + 1, dim)
        # print('layer_position_embeddings shape: ', layer_position_embeddings.shape)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat((cls_tokens, x), dim=1)  # (bs, seq_len + 1, d)
        x = x + position_embeddings
        x = self.layer_norm(x)
        x = self.dropout(x)
        # (batch_size, seq_len + 1, dim) -> (seq_len + 1, batch_size, dim) -> (batch_size, seq_len + 1, dim)
        out = self.encoder(x.permute(1, 0, 2)).permute(1, 0, 2)
        return out[:, 0, :], out[:, 1:, :]

