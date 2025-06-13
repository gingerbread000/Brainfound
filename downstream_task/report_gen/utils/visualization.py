# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/05/07 20:00
# @Author  : Liangdi.Ma
import os

import cv2
import numpy as np

def plot_mim_result(gt, pred, mask=None, name=[], save_dir=''):
    """
    gt, pred, mask: ndarray of (scan num, slice num, c, h, w),
    where mask=0 means visible and mask=1 means masked(what we care)
    name: list of str, path of each scan, length=scan num
    save_dir: str,
    """
    sample_num, im_num, _, _, _ = gt.shape
    if mask is None:
        masked_im = gt
        recon_im = pred
    else:
        masked_im = np.where(mask == 1, 1, gt)
        recon_im = np.where(mask == 1, pred, gt)

    if len(name) == 0:
        name = list(np.arange(sample_num))
    else:
        assert len(name) == sample_num
        name = [','.join([s for s in os.path.splitext(p)[0].split('/') if s != '']) for p in name]

    for i_sample in range(sample_num):
        sample_save_dir = os.path.join(save_dir, str(name[i_sample]))
        os.makedirs(sample_save_dir, exist_ok=True)
        for i_im in range(im_num):
            gt_img = gt[i_sample, i_im]  # (c, h, w)
            masked_img = masked_im[i_sample, i_im]  # (c, h, w)
            pred_img = pred[i_sample, i_im]  # (c, h, w)
            recon_img = recon_im[i_sample, i_im]
            gt_img = np.clip(gt_img * 255, 0, 255)
            masked_img = np.clip(masked_img * 255, 0, 255)
            pred_img = np.clip(pred_img * 255, 0, 255)
            recon_img = np.clip(recon_img * 255, 0, 255)
            tmp_img = np.concatenate([gt_img, masked_img, pred_img, recon_img], axis=-1)  # (c, h, 3w)
            concat_img = []
            for c in range(tmp_img.shape[0]):  # channel wise
                concat_img.append(tmp_img[c])
            concat_img = np.concatenate(concat_img, axis=0)  # (h * c, 2w)
            p = os.path.join(sample_save_dir, f'{i_im + 1}.png')
            cv2.imwrite(p, concat_img)


