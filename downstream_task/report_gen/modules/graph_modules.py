# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/10/09 21:24
# @Author  : Liangdi.Ma

import torchvision.models as models
import torch.nn.functional as F
from torch.nn import Parameter
import torch
import torch.nn as nn
import math

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, edge_weight=None):
        support = torch.matmul(input, self.weight)  # input:(bs, N, d_in), weight: (d_in, d_out), support: (bs, N, d_out)
        if edge_weight is not None:
            adj = torch.mul(adj, edge_weight)  # adj: (N, N)
        output = torch.matmul(adj, support)  # (bs, N, d_out)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True, norm=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.norm = norm

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        if self.norm:
            raise NotImplementedError
            # self.normalize = nn.BatchNorm1d(out_features)

    def forward(self, h, adj, return_attention=False):
        Wh = torch.matmul(h, self.W)  # h.shape: (bs, N, in_features), Wh.shape: (bs, N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)  # (bs, N, N), e(i,j) = score of neighbor node j to target node i

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # (bs, N, N)
        attention = F.softmax(attention, dim=-1)  # (bs, N, N), !!!modify as dim=2 (original: dim=1)
        attn = self.dropout(attention)
        h_prime = torch.matmul(attn, Wh)  # (bs, N, N) * (bs, N, d_out) -> (bs, N, d_out)

        # print('adj shape: ', adj.shape)
        # print('attention shape: ', attention.shape)
        # print('Wh shape: ', Wh.shape)
        # exit()

        # if self.norm:
        #     h_prime = self.normalize(h_prime)

        if self.concat:
            out = F.elu(h_prime)
        else:
            out = h_prime

        if return_attention:
            return out, attention
        else:
            return out


    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (bs, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (bs, N, 1)
        # e.shape (bs, N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class DynamicGraphAttentionLayer(nn.Module):
    """
    GATv2 layer, from paper: How Attentive are Graph Attention Networks?
    see:
    https://arxiv.org/abs/2105.14491
    """

    def __init__(self, in_features, out_features,  dropout=0.1, alpha=0.2, concat=True, norm=False):
        super(DynamicGraphAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.norm = norm

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        if self.norm:
            raise NotImplementedError
            # self.normalize = nn.BatchNorm1d(out_features)

    def forward(self, h, adj, return_attention=False):
        # h.shape: (bs, N, in_features)
        # note: A(n, 2d_in) * B(2d_in, d_out) = [A1(n, d_in), A2(n, d_in)] * [B1(d_in, d_out) || B2(d_in, d_out)]
        # = A1(n, d_in) * B1(d_in, d_out) + A2(n, d_in) * B2(d_in, d_out)
        # = C(n, d_out)
        wh = torch.matmul(h, self.W)  # (bs, N, out_feature)
        e = self._prepare_attentional_mechanism_input(wh)  # (bs, N, N), e(i,j) = score of neighbor node j for node i

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # (bs, N, N)
        attention = F.softmax(attention, dim=-1)  # (bs, N, N)
        attn = self.dropout(attention)  # (bs, N, N)
        h_prime = torch.matmul(attn, wh)

        if self.concat:
            out = F.elu(h_prime)
        else:
            out = h_prime

        if return_attention:
            return out, attention
        else:
            return out

    def _prepare_attentional_mechanism_input(self, wh):
        # h.shape (bs, N, in_feature)
        # self.W.shape (in_feature, out_feature)
        # self.a.shape (out_feature, 1)
        # whi = torch.matmul(h, self.Wi)  # (bs, N, out_feature)
        # whj = torch.matmul(h, self.Wj)  # (bs, N, out_feature)
        wh_c = wh.unsqueeze(2) + wh.unsqueeze(1)  # (bs, N, N, out_feature)
        wh_a = self.leakyrelu(wh_c)  # (bs, N, N, out_feature)
        e = torch.matmul(wh_a, self.a).squeeze(-1)  # (bs, N, N)
        return e

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GraphSAGELayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.in_features = in_features
        self.out_features = out_features
        # self.fc_neigh = nn.Linear(in_features, out_features)
        # self.fc_self = nn.Linear(in_features, out_features)
        self.fc_concat = nn.Linear(2 * in_features, out_features)
        self.reset_parameters()
        self.relu = nn.ReLU()

    def reset_parameters(self):
        r"""
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_concat.weight, gain=gain)
        # nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        # nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, h, edge_weight):
        # h.shape: (bs, N, in_features), edge_weight: (N, N) with diag as 0, transpose matrix of adj
        # drop_h = self.dropout(h)
        # feat_neigh = self.fc_neigh(drop_h)  # (bs, N, out_features)
        # feat_self = self.fc_self(drop_h)  # (bs, N, out_features)
        aggregate_neigh = torch.matmul(edge_weight, h)  # (bs, N, in_feature)
        norm_aggregate_neigh = aggregate_neigh / (torch.sum(edge_weight, dim=-1, keepdim=True) + 1e-6)
        feat_concat = torch.cat([h, norm_aggregate_neigh], dim=-1)  # (bs, N, 2 * in_features)
        drop_feat_concat = self.dropout(feat_concat)
        h_prime = self.relu(self.fc_concat(drop_feat_concat))  # (bs, N, out_features)
        # norm_h_prime = h_prime / (torch.norm(h_prime, p=2, dim=-1, keepdim=True) + 1e-6)

        return h_prime


class MyGraphLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(MyGraphLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.fc_neigh = nn.Linear(in_features, out_features)
        self.fc_self = nn.Linear(in_features, out_features)
        self.fc_concat = nn.Linear(2 * out_features, out_features)
        self.reset_parameters()
        self.relu = nn.ReLU()

    def reset_parameters(self):
        r"""
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_concat.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, h, edge_weight):
        # h.shape: (bs, N, in_features), edge_weight: (N, N) with diag as 0, transpose matrix of adj
        drop_h = self.dropout(h)
        feat_neigh = self.relu(self.fc_neigh(drop_h))  # (bs, N, out_features)
        aggregate_neigh = torch.matmul(edge_weight, feat_neigh)  # (bs, N, out_features)
        norm_aggregate_neigh = aggregate_neigh / (torch.sum(edge_weight, dim=-1, keepdim=True) + 1e-6)
        # if do not apply norm to aggregate features, the node with much neighbors and few neighbors will be diversity
        feat_self = self.relu(self.fc_self(drop_h))  # (bs, N, out_features)
        feat_concat = torch.cat([feat_self, norm_aggregate_neigh], dim=-1)  # (bs, N, 2 * out_features)
        drop_feat_concat = self.dropout(feat_concat)
        h_prime = self.relu(self.fc_concat(drop_feat_concat))  # (bs, N, out_features)
        # norm_h_prime = h_prime / (torch.norm(h_prime, p=2, dim=-1, keepdim=True) + 1e-6)

        return h_prime
