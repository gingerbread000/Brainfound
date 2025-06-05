# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/08/08 20:14
# @Author  : Liangdi.Ma
import torch
import torch.nn as nn

class DenseModel(nn.Module):
    def __init__(self, in_features, out_features=32, activation='relu'):
        super(DenseModel, self).__init__()
        self.output_hidden_size = out_features
        if self.output_hidden_size == 32:
            self.backbone = nn.Sequential(
                nn.Linear(in_features, 64),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
        elif self.output_hidden_size == 1024:
            self.backbone = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(2048, 1024),
                nn.ReLU(),
            )
        elif self.output_hidden_size == 512:
            self.backbone = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(2048, 512),
                nn.ReLU(),
            )
        # elif self.output_hidden_size == 768:
        #     self.backbone = nn.Sequential(
        #         nn.Linear(in_features, 256),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.1),
        #         nn.Linear(256, 512),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.1),
        #         nn.Linear(512, 1024),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.2),
        #         nn.Linear(1024, 768),
        #     )

        elif self.output_hidden_size == 768:
            self.backbone = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 768),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError('Unsupport output dim')

        # self.dropout = nn.Dropout(p=0.1)
        # self.cls_head = nn.Sequential(
        #     nn.Linear(self.hidden_feature_size, self.hidden_feature_size),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_feature_size, num_classes),
        # )
    
    def forward(self, x, missing_mask=None):
        y = self.backbone(x)
        # y = self.dropout(y)
        # y = self.cls_head(y)
        return y, 0

class BiomarkerAttention(nn.Module):
    def __init__(self, bio_num, hidden_dim=256, nlayers=2, nheads=8, feedforward_dim=1024, dropout=0.1):
        super(BiomarkerAttention, self).__init__()
        self.bio_embedding = torch.nn.Parameter(torch.randn((bio_num, hidden_dim)))  # (N, d)
        self.cls_embedding = torch.nn.Parameter(torch.randn((1, hidden_dim)))  # (N, d)
        _layer = nn.TransformerEncoderLayer(
            hidden_dim,
            nheads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(_layer, nlayers)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-8, elementwise_affine=True)
        self.apply(self._init_weights)
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.output_hidden_size = hidden_dim

    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, x, missing_mask=None):
        """
        x: input biomarker vector, (bs, N)
        mask(binary matrix): indicate which is missing value, (bs, N):
            0 means ground truth, 1 means missing
        """
        assert missing_mask is not None, "the mask of missing lab test value is not specified."
        batch_size, bio_num = x.shape
        # expand each test value into a test feature: (bs, N) * (N, d) -> (bs, N, d)
        bio_embeddings = torch.einsum('bi,ij->bij', x, self.bio_embedding)

        # add global embedding to input and mask, and take layer norm: (bs, N+1, d)
        cls_embedding = torch.stack([self.cls_embedding]*batch_size, dim=0)
        input_embeddings = torch.cat([cls_embedding, bio_embeddings], dim=1)
        input_embeddings = self.layer_norm(input_embeddings)
        missing_mask = torch.cat([torch.zeros(batch_size, 1, device=missing_mask.device), missing_mask], dim=-1)  # (bs, N+1)

        # transpose input: (N+1, bs, d)
        input_embeddings = input_embeddings.transpose(0, 1)
        missing_mask = (missing_mask != 0)

        # output: (N+1, bs, d) -> (bs, N+1, d)
        output_embeddings = self.transformer_encoder(input_embeddings, src_key_padding_mask=missing_mask)
        output_embeddings = output_embeddings.transpose(0, 1)

        return output_embeddings[:, 0, ...], output_embeddings[:, 1:, ...]


class WideLayer(nn.Module):
    def __init__(self):
        super(WideLayer, self).__init__()

    def forward(self, wide_feature):
        return wide_feature

class DeepLayer(nn.Module):
    def __init__(self, in_features):
        super(DeepLayer,self).__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.dense3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.hidden_feature_size = 64
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, deep_feature):
        x = self.dense1(deep_feature)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dropout(x)
        return x

class WideAndDeepModel(nn.Module):
    def __init__(self, in_features, num_classes):
        super(WideAndDeepModel, self).__init__()
        self.wide_module = WideLayer()
        self.deep_module = DeepLayer(in_features=in_features)
        self.wide_projector = nn.Linear(in_features, num_classes, bias=False)
        self.deep_projector = nn.Linear(self.deep_module.hidden_feature_size, num_classes, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros((1, 2)))

    def forward(self, x):
        wide_feature = self.wide_module(x)
        deep_feature = self.deep_module(x)
        wide_output = self.wide_projector(wide_feature)
        deep_output = self.deep_projector(deep_feature)
        output = wide_output + deep_output + self.bias
        return output

class CrossNet(nn.Module):
    def __init__(self, in_features, num_classes, use_softmax=False):
        super(CrossNet, self).__init__()
        self.cross_layer0 = nn.Linear(in_features, 1)
        self.cross_layer1 = nn.Linear(in_features, 1)
        self.deep_layer0 = nn.Linear(3 * in_features, 2 * in_features)
        self.deep_layer1 = nn.Linear(2 * in_features, in_features)
        self.deep_layer2 = nn.Linear(in_features, num_classes)
        self.activation = nn.ReLU()
        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dense_x):
        dense_x = dense_x.unsqueeze(-1)  # (bs, n, 1)
        cross_x0 = torch.bmm(dense_x, dense_x.transpose(1, 2))  # (bs, n, n)
        cross_f1 = self.cross_layer0(cross_x0)  # (bs, n, 1)
        cross_x1 = torch.bmm(dense_x, cross_f1.transpose(1, 2))  # (bs, n, n)
        cross_f2 = self.cross_layer1(cross_x1)  # (bs, n, 1)
        deep_x0 = torch.cat([dense_x, cross_f1, cross_f2], dim=1).squeeze(-1)  # (bs, 3n, 1) -> (bs, 3n)
        deep_x1 = self.activation(self.deep_layer0(deep_x0))  # (bs, 2n)
        deep_x2 = self.activation(self.deep_layer1(deep_x1))  # (bs, n)
        output = self.deep_layer2(deep_x2)  # (bs, class)
        if self.use_softmax:
            output = self.softmax(output)
        return output





