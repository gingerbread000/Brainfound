import time
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve


class Meter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TimeMeter(Meter):
    def __init__(self):
        Meter.__init__(self)

    def start(self):
        self.start_time = time.time()

    def record(self, n = 1):
        spent_time = time.time() - self.start_time
        self.update(spent_time, n)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        if isinstance(output, list) and isinstance(target, list):
            assert len(output) == len(target)
            corr = 0
            total = len(target)
            for l in range(len(output)):
                if output[l] == target[l]:
                    corr += 1
            return corr * 100 / total

        if output.shape[1] == 1:
            output = (output.view(-1) > 0.5).long()
            correct = output.eq(target.view(-1))
            return [torch.sum(correct).float() / correct.shape[0] * 100]

        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def precision(c_matrix,ti):
    pre = c_matrix[ti,ti] / np.sum(c_matrix[:,ti])
    return pre

def recall(c_matrix,ti):
    recall = c_matrix[ti,ti] / np.sum(c_matrix[ti])
    return recall

def f_score(c_matrix,ti):
    pre = c_matrix[ti, ti] / np.sum(c_matrix[:, ti])
    recall = c_matrix[ti, ti] / np.sum(c_matrix[ti])
    score = 2 * pre * recall / (pre + recall)
    return score

def comfusion_matrix(preds, labels, c_num=2):
    confuse_m = np.zeros((c_num, c_num))
    for i in range(len(labels)):
        label = int(labels[i])
        pred = int(preds[i])
        confuse_m[label, pred] += 1
    return confuse_m

# def comfusion_matrix(output, target, c_num, thred = 0.5):
#     confuse_m = np.zeros((c_num, c_num))
#     if len(output.shape) == 1:
#         pred = output
#     else:
#         _, pred = torch.max(output, dim=1)
#
#     for m in range(pred.shape[0]):
#         label = target[m].item()
#         label = int(label)
#
#         if len(output.shape) == 1:
#             if output[m] > thred:
#                 pre = 1
#             else:
#                 pre = 0
#
#         elif output.shape[1] == 1:
#             if output[m][0] > thred:
#                 pre = 1
#             else:
#                 pre = 0
#         else:
#             pre = pred[m].item()
#
#
#         confuse_m[label][pre] += 1
#     return confuse_m
#
def auc_score(y_true,y_scores):
    if isinstance(y_true,torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    if isinstance(y_true, type([])):
        y_true = np.array(y_true).reshape((-1))

    if isinstance(y_scores, type([])):
        y_scores = np.array(y_scores).reshape((-1))

    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = -1
    return auc

def ap_score(y_true,y_scores):
    if isinstance(y_true,torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    if isinstance(y_true, type([])):
        y_true = np.array(y_true).reshape((-1))

    if isinstance(y_scores, type([])):
        y_scores = np.array(y_scores).reshape((-1))

    try:
        ap = average_precision_score(y_true, y_scores)
    except:
        ap = -1
    return ap

def roc(y_true,y_scores):
    if isinstance(y_true,torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    if isinstance(y_true, list):
        y_true = np.array(y_true).reshape((-1))

    if isinstance(y_scores, list):
        y_scores = np.array(y_scores).reshape((-1))

    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        sen = tpr
        spe = 1 - fpr

        m_list = sen * spe * 2 / (sen + spe)
        m_ind = np.argmax(m_list)

        return sen[m_ind], spe[m_ind]
    except:
        return -1, -1

def best_thred(y_true,y_scores):
    if isinstance(y_true,torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    if isinstance(y_true, list):
        y_true = np.array(y_true).reshape((-1))

    if isinstance(y_scores, list):
        y_scores = np.array(y_scores).reshape((-1))

    try:
        fpr, tpr, thred = roc_curve(y_true, y_scores)
        sen = tpr
        spe = 1 - fpr
        m_list = np.array(sen * spe * 2 / (sen + spe))
        idx = np.argmax(m_list)
        # idx = np.argmax(tpr - fpr)
        return thred[idx]
    except:
        return -1
