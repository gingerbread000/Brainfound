# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/06/23 21:04
# @Author  : Liangdi.Ma

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def auc_score(y_true, y_scores):
    if isinstance(y_true,torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    if isinstance(y_true, type([])):
        y_true = np.array(y_true)

    if isinstance(y_scores, type([])):
        y_scores = np.array(y_scores)

    assert len(y_true.shape) == 2

    auc_list = []
    for i in range(y_true.shape[-1]):
        tmp_y_true = y_true[:, i]
        tmp_y_scores = y_scores[:, i]
        try:
            tmp_auc = roc_auc_score(tmp_y_true, tmp_y_scores)
        except:
            tmp_auc = -1
        auc_list.append(tmp_auc)
    auc_avg = np.mean(auc_list)
    return auc_avg, auc_list

# def compute_multi_label_metrics(y_true, y_scores, thred=0.5):
#     # y_true: (N, c), y_scores: (N, c)
#     if isinstance(y_true, torch.Tensor):
#         y_true = y_true.detach().cpu().numpy()
#
#     if isinstance(y_scores, torch.Tensor):
#         y_scores = y_scores.detach().cpu().numpy()
#
#     if isinstance(y_true, type([])):
#         y_true = np.array(y_true)
#
#     if isinstance(y_scores, type([])):
#         y_scores = np.array(y_scores)
#
#     assert len(y_true.shape) == 2
#     y_preds = np.where(y_scores >= thred, 1, 0)
#
#     v_metrics = {}
#
#     auc_scores = roc_auc_score(y_true, y_scores, average=None)
#     if isinstance(auc_scores, np.float64):
#         v_metrics['auc'] = [auc_scores]
#     else:
#         v_metrics['auc'] = list(auc_scores)
#
#     ap_scores = average_precision_score(y_true, y_scores, average=None)
#     if isinstance(ap_scores, np.float64):
#         v_metrics['ap'] = [ap_scores]
#     else:
#         v_metrics['ap'] = list(ap_scores)
#
#     prec_scores = precision_score(y_true, y_preds, average=None)
#     if isinstance(prec_scores, np.float64):
#         v_metrics['precision'] = [prec_scores]
#     else:
#         v_metrics['precision'] = list(prec_scores)
#
#     rec_scores = recall_score(y_true, y_preds, average=None)
#     if isinstance(rec_scores, np.float64):
#         v_metrics['recall'] = [rec_scores]
#     else:
#         v_metrics['recall'] = list(rec_scores)
#
#     f1_scores = f1_score(y_true, y_preds, average=None)
#     if isinstance(f1_scores, np.float64):
#         v_metrics['f1'] = [f1_scores]
#     else:
#         v_metrics['f1'] = list(f1_scores)
#
#     v_metrics['macro-f1'] = [sum(v_metrics['f1'])/len(v_metrics['f1'])]
#     v_metrics['micro-f1'] = [f1_score(y_true, y_preds, average='micro')]
#     # v_metrics['map'] = [sum(v_metrics['ap'])/len(v_metrics['ap'])]
#
#     return v_metrics

def predict_topk(scores, k=3):
    # 取每个样本的top-k个预测结果！
    # array in (N, C)
    n_sample, n_class = scores.shape
    preds_scalar = np.argsort(scores, axis=-1)[:, -k:]
    preds = np.zeros_like(scores)  # [n_sample, n_class]
    for sample in range(n_sample):
        for c in preds_scalar[sample]:
            preds[sample, c] = 1
    return preds


def compute_multi_label_metrics(y_true, y_scores, thred=0.5, topk=-1, return_test=False):
    # y_true: (N, c), y_scores: (N, c), multi-hot matrix
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    if isinstance(y_true, type([])):
        y_true = np.array(y_true)

    if isinstance(y_scores, type([])):
        y_scores = np.array(y_scores)

    assert len(y_true.shape) == 2
    n_sample, n_class = y_true.shape
    if topk > 0:
        y_preds = predict_topk(y_scores, topk)  # [n_sample, n_class]
    else:
        y_preds = np.where(y_scores >= thred, 1, 0)  # array, (n, c)

    v_metrics = {}

    auc_scores = roc_auc_score(y_true, y_scores, average=None)
    if isinstance(auc_scores, np.float64):
        v_metrics['auc'] = [auc_scores]
    else:
        v_metrics['auc'] = list(auc_scores)

    f1_scores = f1_score(y_true, y_preds, average=None)
    if isinstance(f1_scores, np.float64):
        v_metrics['f1'] = [f1_scores]
    else:
        v_metrics['f1'] = list(f1_scores)

    intersection_numerator = (y_true + y_preds) // 2
    union_denominator = np.maximum(y_true, y_preds)
    jaccard_score = np.mean(np.sum(intersection_numerator, axis=-1) / np.sum(union_denominator, axis=-1))
    v_metrics['jaccard'] = [jaccard_score]

    ap_scores = average_precision_score(y_true, y_scores, average=None)
    if isinstance(ap_scores, np.float64):
        v_metrics['ap'] = [ap_scores]
    else:
        v_metrics['ap'] = list(ap_scores)

    prec_scores = precision_score(y_true, y_preds, average=None)
    if isinstance(prec_scores, np.float64):
        v_metrics['precision'] = [prec_scores]
    else:
        v_metrics['precision'] = list(prec_scores)

    rec_scores = recall_score(y_true, y_preds, average=None)
    if isinstance(rec_scores, np.float64):
        v_metrics['recall'] = [rec_scores]
    else:
        v_metrics['recall'] = list(rec_scores)

    # for k in [6, 8, 10, 12, 15, 20]:
    #     tmp_preds = predict_topk(y_scores, k)  # [n_sample, n_class]
    #     rec_topk = recall_score(y_true, tmp_preds, average=None)
    #     if isinstance(rec_scores, np.float64):
    #         v_metrics[f'recall@{k}'] = [rec_topk]
    #     else:
    #         v_metrics[f'recall@{k}'] = list(rec_topk)

    v_metrics_sample = {}
    auc_scores_sample = roc_auc_score(y_true, y_scores, average='samples')
    if isinstance(auc_scores_sample, np.float64):
        v_metrics_sample['auc'] = [auc_scores_sample]
    else:
        v_metrics_sample['auc'] = list(auc_scores_sample)
    f1_scores_sample = f1_score(y_true, y_preds, average='samples')
    if isinstance(f1_scores_sample, np.float64):
        v_metrics_sample['f1'] = [f1_scores_sample]
    else:
        v_metrics_sample['f1'] = list(f1_scores_sample)
    intersection_numerator = (y_true + y_preds) // 2
    union_denominator = np.maximum(y_true, y_preds)
    jaccard_score_sample = np.mean(np.sum(intersection_numerator, axis=-1) / np.sum(union_denominator, axis=-1))
    v_metrics_sample['jaccard'] = [jaccard_score_sample]

    ap_scores_sample = average_precision_score(y_true, y_scores, average='samples')
    if isinstance(ap_scores_sample, np.float64):
        v_metrics_sample['ap'] = [ap_scores_sample]
    else:
        v_metrics_sample['ap'] = list(ap_scores_sample)

    prec_scores_sample = precision_score(y_true, y_preds, average='samples')
    if isinstance(prec_scores_sample, np.float64):
        v_metrics_sample['precision'] = [prec_scores_sample]
    else:
        v_metrics_sample['precision'] = list(prec_scores_sample)

    rec_scores_sample = recall_score(y_true, y_preds, average='samples')
    if isinstance(rec_scores_sample, np.float64):
        v_metrics_sample['recall'] = [rec_scores_sample]
    else:
        v_metrics_sample['recall'] = list(rec_scores_sample)

    topk_list = [k for k in [6, 8, 10, 12, 15, 20] if k <= n_class]
    for k in topk_list:
        tmp_preds = predict_topk(y_scores, k)  # [n_sample, n_class]
        rec_topk = recall_score(y_true, tmp_preds, average=None)
        rec_topk_sample = recall_score(y_true, tmp_preds, average='samples')
        if isinstance(rec_topk, np.float64):
            v_metrics[f'recall@{k}'] = [rec_topk]
        else:
            v_metrics[f'recall@{k}'] = list(rec_topk)
        if isinstance(rec_topk_sample, np.float64):
            v_metrics_sample[f'recall@{k}'] = [rec_topk_sample]
        else:
            v_metrics_sample[f'recall@{k}'] = list(rec_topk_sample)

    if return_test:
        avg_med_num = np.mean(np.sum(y_preds, axis=-1))  # (N,)
        v_metrics_sample.update({'med_num': avg_med_num})
        return v_metrics_sample

    return v_metrics, v_metrics_sample

if __name__ == '__main__':
    targets = torch.from_numpy(np.array([[1,1,0,0,0,1,0,1,0,1],[1,0,1,0,1,0,1,0,1,1],[0,1,1,1,1,1,0,0,0,0]]))
    scores = torch.rand((3, 10))
    # targets = torch.from_numpy(np.array([[1], [1], [0]]))
    # scores = torch.rand((3, 1))
    # print('target: ')
    # print(targets)
    # print('scores: ')
    # print(scores)
    _,v_metric = compute_multi_label_metrics(targets, scores)
    print(v_metric)