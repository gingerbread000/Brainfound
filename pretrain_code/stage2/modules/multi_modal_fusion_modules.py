# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/08/15 14:59
# @Author  : Liangdi.Ma
import torch
import torch.nn as nn

class ConcatFusion(nn.Module):
    def __init__(self, concat_dim=1568, output_dim=256):
        super(ConcatFusion, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )
        self.output_hidden_size = output_dim

    def forward(self, feature_tuple):
        concat = torch.cat(list(feature_tuple), dim=-1)
        y = self.mlp(concat)
        return y, None

class ConcatPFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=128):
        super(ConcatPFusion, self).__init__()
        if isinstance(input_dim, tuple):
            input_dim = list(input_dim)
        self.input_num = len(input_dim)
        self.projection = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim[i], input_dim[i]),
                    nn.ReLU(),
                    nn.Linear(input_dim[i], hidden_dim)
                )
                for i in range(len(input_dim))]
        )
        in_feature = len(input_dim) * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_feature, in_feature),
            nn.ReLU(),
            nn.Linear(in_feature, output_dim),
            nn.ReLU(),
        )
        self.output_hidden_size = output_dim

    def forward(self, feature_tuple):
        split_hidden_state = [self.projection[i](feature_tuple[i]) for i in range(self.input_num)]  # [(bs, 128)*3]
        concat = torch.cat(split_hidden_state, dim=-1)  # (bs, 128*3)
        y = self.mlp(concat)
        return y, None

# class CrossAttentionFusion(nn.Module):
#     def __init__(self):
#         super(CrossAttentionFusion, self).__init__()
#
#     def forward(self):
#         pass

class AttentionFusion(nn.Module):
    def __init__(self, input_dim, nlayers=2, hidden_dim=128, nheads=4, feedforward_dim=512, dropout=0.1):
        super(AttentionFusion, self).__init__()
        if isinstance(input_dim, tuple):
            input_dim = list(input_dim)
        self.input_num = len(input_dim)
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim[i], input_dim[i]),
                    nn.ReLU(),
                    nn.Linear(input_dim[i], hidden_dim)
                )
                for i in range(len(input_dim))]
        )

        LayerClass = (
            nn.TransformerEncoderLayer
        )
        try:
            _layer = LayerClass(
                hidden_dim,
                nheads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                attn_drop=nn.Dropout(p=dropout),
            )
        except:
            _layer = LayerClass(
                hidden_dim,
                nheads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
            )
        self.transformer_encoder = nn.TransformerEncoder(_layer, nlayers)
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

    def forward(self, feature_tuple):
        split_hidden_state = [self.mlp[i](feature_tuple[i]) for i in range(self.input_num)]  # [(bs, d)*3]
        concat_input = torch.stack(split_hidden_state, dim=0)  # (3, bs, d)
        # print('input shape: ', concat_input.shape)
        output = self.transformer_encoder(concat_input)  # (l, batch_size, hidden_size)
        # Undo the transpose and bring batch to dim 0.
        output = torch.squeeze(self.avgpool1d(output.permute(1, 2, 0)), dim=-1)  # (batch_size, max_caption_length, vocab_size)
        return output, 0

class FeatWeightedFusion(nn.Module):
    def __init__(self, input_dim, output_dim, target='index'):
        super(FeatWeightedFusion, self).__init__()
        self.target = target
        if isinstance(input_dim, tuple):
            input_dim = list(input_dim)
        self.input_dim = input_dim
        self.intra_norm_pre = nn.LayerNorm(sum(input_dim))  # nn.ModuleList([nn.LayerNorm(input_dim[i]) for i in range(len(input_dim))])
        self.intra_norm_post = nn.LayerNorm(sum(input_dim))
        self.attn_mlp = nn.Sequential(
            nn.Linear(sum(input_dim), sum(input_dim) // 2),
            nn.GELU(),
            nn.Linear(sum(input_dim) // 2, sum(input_dim)),
        )
        self.fusion_mlp = nn.Sequential(
                nn.Linear(sum(input_dim), 512),
                nn.GELU(),
                nn.Linear(512, sum(input_dim)),
            )
        self.output_hidden_size = sum(input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature_tuple):
        input = torch.cat(feature_tuple, dim=-1)
        norm_features = self.intra_norm_pre(input)
        ########## ??????????? ###########
        y_h = self.softmax(self.attn_mlp(norm_features))  # (bs, n) or (bs, 3)
        y = y_h * norm_features + norm_features
        skip_y = y + input
        out_norm = self.intra_norm_post(skip_y)
        out_h = self.fusion_mlp(out_norm)
        out = out_h * out_norm + out_norm + skip_y
        return out, y_h

# class ModalWeightedFusion(nn.Module):
#     def __init__(self, input_dim, output_dim, target='index'):
#         super(ModalWeightedFusion, self).__init__()
#         self.target = target
#         if isinstance(input_dim, tuple):
#             input_dim = list(input_dim)
#         self.input_dim = input_dim
#         self.intra_norm_pre = nn.ModuleList([nn.BatchNorm1d(input_dim[i]) for i in range(len(input_dim))])
#         self.intra_norm_post = nn.ModuleList([nn.LayerNorm(input_dim[i]) for i in range(len(input_dim))])
#
#         self.attn_mlp = nn.Sequential(
#             nn.Linear(sum(input_dim), sum(input_dim) // 2),
#             nn.GELU(),
#             nn.Linear(sum(input_dim) // 2, len(input_dim)),
#         )
#
#         self.fusion_mlp = nn.Sequential(
#                 nn.Linear(sum(input_dim), 512),
#                 nn.GELU(),
#                 nn.Linear(512, sum(input_dim)),
#             )
#         self.output_hidden_size = sum(input_dim)
#
#     def forward(self, feature_tuple):
#         input = torch.cat(feature_tuple, dim=-1)
#         norm_features = self.intra_norm_pre(input)
#         y = self.attn_mlp(norm_features)  # (bs, n) or (bs, 3)
#         y_h = [y[:, i:i+1] * norm_features[:, i] for i in range(len(self.input_dim))]  # list of weights
#         y_h = torch.cat(y_h, dim=-1) + norm_features
#         skip_y_h = y_h + input
#         y_h_norm = self.intra_norm_post(skip_y_h)
#         out = self.fusion_mlp(y_h_norm) * y_h_norm + y_h_norm + skip_y_h
#         return out, y_h

class ModalWeightedFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=128, fusion='sum'):
        super(ModalWeightedFusion, self).__init__()
        if isinstance(input_dim, tuple):
            input_dim = list(input_dim)
        self.projection = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim[i], input_dim[i]),
                    nn.ReLU(),
                    nn.Linear(input_dim[i], hidden_dim)
                )
                for i in range(len(input_dim))]
        )
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.input_num = len(self.input_dim)
        self.output_hidden_size = output_dim
        self.intra_norm_pre = nn.BatchNorm1d(self.input_num * hidden_dim)  # nn.ModuleList([nn.LayerNorm(input_dim[i]) for i in range(len(input_dim))])
        # self.intra_norm_post = nn.BatchNorm1d(sum(input_dim))
        self.attn_mlp = nn.Sequential(
            nn.Linear(self.input_num * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_num),
        )
        self.fusion = fusion
        if self.fusion == 'sum':
            self.fusion_dim = hidden_dim
        else:
            self.fusion_dim = self.input_num * hidden_dim

        self.fusion_mlp = nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(self.fusion_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature_tuple):
        split_hidden_state = [self.projection[i](feature_tuple[i]) for i in range(self.input_num)]  # [(bs, 128)*3]
        concat = torch.cat(split_hidden_state, dim=-1)  # (bs, 128*3)
        norm_features = self.intra_norm_pre(concat)  # (bs, 128*3)
        w_h = self.softmax(self.attn_mlp(norm_features))  # (bs, 3)
        y_h = [w_h[:, i:i+1] * torch.split(norm_features, self.hidden_dim, dim=-1)[i] for i in range(self.input_num)]
        if self.fusion == 'sum':
            y = torch.sum(torch.stack(y_h, dim=-1), dim=-1)  # (bs, 128, 3) -> (bs, 128)
        else:
            y = torch.cat(y_h, dim=-1)  # + norm_features  # (bs, 128*3)
        if self.fusion_mlp is not None:
            out = self.fusion_mlp(y)
        else:
            out = y

        return out, w_h

# class WeightedFusion(nn.Module):
#     def __init__(self, input_dim, output_dim=1024, target='index'):
#         super(WeightedFusion, self).__init__()
#         self.target = target
#         if isinstance(input_dim, tuple):
#             input_dim = list(input_dim)
#         self.output_hidden_size = sum(input_dim)
#         # self.intra_norm = [nn.LayerNorm(input_dim[i]) for i in range(len(input_dim))]
#         if self.target == 'index':
#             self.mlp = nn.Sequential(
#                 nn.Linear(sum(input_dim), sum(input_dim) // 2),
#                 nn.ReLU(),
#                 nn.Linear(sum(input_dim) // 2, sum(input_dim)),
#             )
#         elif self.target == 'modal':
#             self.mlp = nn.Sequential(
#                 nn.Linear(sum(input_dim), sum(input_dim) // 2),
#                 nn.ReLU(),
#                 nn.Linear(sum(input_dim) // 2, len(input_dim)),
#             )
#         else:
#             raise NotImplementedError
#
#     def forward(self, feature_tuple):
#         # norm_features = [self.intra_norm[i](feature_tuple[i]) for i in range(len(feature_tuple))]
#         concat = torch.cat(feature_tuple, dim=-1)  # (bs, n)
#         y = self.mlp(concat)  # (bs, n) or (bs, 3)
#         if self.target == 'index':
#             y_h = y * concat + concat
#         elif self.target == 'modal':
#             y_h = [y[:, i:i+1] * feature_tuple[i] for i in range(len(feature_tuple))]  # list of weights
#             y_h = torch.cat(y_h, dim=-1) + concat
#         else:
#             raise NotImplementedError
#         return y_h, y

class PredictVotingFusion(nn.Module):
    def __init__(self, input_dim, class_num=1, net='voting'):
        super(PredictVotingFusion, self).__init__()
        if isinstance(input_dim, tuple):
            input_dim = list(input_dim)
        self.input_dim = input_dim
        self.input_num = len(input_dim)
        self.class_num = class_num
        self.net = net
        cls_heads = []
        for dim in self.input_dim:
            tmp_head = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, self.class_num),
            )
            cls_heads.append(tmp_head)
        self.cls_heads = nn.ModuleList(cls_heads)
        if self.net == 'classifier1d':
            self.pred_fusion_heads = nn.Sequential(
                nn.Linear(self.input_num * self.class_num, self.class_num),
            )
        elif self.net == 'classifier2d':
            self.pred_fusion_heads = nn.ModuleList([
                nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(self.class_num, self.class_num,)),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Linear(4, self.class_num),
            ])

    def forward(self, feature_tuple):
        split_predictions = [torch.sigmoid(self.cls_heads[i](feature_tuple[i])) for i in range(self.input_num) if
                             feature_tuple[i] is not None]  # [(bs, c)*k] where k can be 1 to 3 (allow any modal miss)
        combine_predictions = torch.stack(split_predictions, dim=-1)  # (bs, c, 3)
        batch_size = combine_predictions.shape[0]
        if self.net == 'voting':
            voting_predictions = torch.mean(combine_predictions, dim=-1)  # (bs, c)
        elif self.net == 'classifier1d':
            voting_predictions = self.pred_fusion_heads(combine_predictions.reshape(batch_size, -1))  # net((N, 3*c))
            voting_predictions = torch.sigmoid(voting_predictions)
        elif self.net == 'classifier2d':
            out = torch.bmm(combine_predictions, combine_predictions.transpose(-2, -1)).unsqueeze(1)
            for k, layer in enumerate(self.pred_fusion_heads):
                out = layer(out)
                if k == 1:
                    out = out.reshape(batch_size, -1)
            voting_predictions = out
            voting_predictions = torch.sigmoid(voting_predictions)
        else:
            raise NotImplementedError
        return voting_predictions, None

class TwoStagePredictVotingFusion(nn.Module):
    def __init__(self, input_num=3, class_num=1, net='voting'):
        super(TwoStagePredictVotingFusion, self).__init__()
        self.input_num = input_num
        self.class_num = class_num
        self.net = net
        if self.net == 'classifier1d':
            self.pred_fusion_heads = nn.Sequential(
                nn.Linear(self.input_num * self.class_num, self.class_num),
            )
        elif self.net == 'classifier2d':
            self.pred_fusion_heads = nn.ModuleList([
                nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(self.class_num, self.class_num,)),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Linear(4, self.class_num),
            ])

    def forward(self, logits_tuple):
        combine_predictions = torch.stack(list(logits_tuple), dim=-1)  # (bs, c, 3)
        batch_size = combine_predictions.shape[0]
        if self.net == 'voting':
            voting_predictions = torch.mean(combine_predictions, dim=-1)  # (bs, c)
        elif self.net == 'classifier1d':
            voting_predictions = self.pred_fusion_heads(combine_predictions.reshape(batch_size, -1))  # net((N, 3*c))
            voting_predictions = torch.sigmoid(voting_predictions)
        elif self.net == 'classifier2d':
            out = torch.bmm(combine_predictions, combine_predictions.transpose(-2, -1)).unsqueeze(1)
            for k, layer in enumerate(self.pred_fusion_heads):
                out = layer(out)
                if k == 1:
                    out = out.reshape(batch_size, -1)
            voting_predictions = out
            voting_predictions = torch.sigmoid(voting_predictions)
        else:
            raise NotImplementedError
        return voting_predictions, None
