# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 20:04
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com


import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class berHuLoss(nn.Module):
    def __init__(self):
        super(berHuLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        huber_c = torch.max(pred - target)
        huber_c = 0.2 * huber_c

        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        diff = diff.abs()

        huber_mask = (diff > huber_c).detach()

        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2

        self.loss = torch.cat((diff, diff2)).mean()

        return self.loss


class ordLoss(nn.Module):
    """
    Ordinal loss is defined as the average of pixelwise ordinal loss F(h, w, X, O)
    over the entire image domain:
    """
    def __init__(self):
        super(ordLoss, self).__init__()
        self.loss = 0.0

    def forward(self, ord_labels, target):
        """
        :param ord_labels: ordinal labels for each position of Image I.
        :param target:     the ground_truth discreted using SID strategy.
        :return: ordinal loss
        """
        # assert pred.dim() == target.dim()
        N, C, H, W = ord_labels.size()
        ord_num = C
        # print('ord_num = ', ord_num)

        self.loss = 0.0

        for k in range(ord_num):
            '''
            p^k_(w, h) = e^y(w, h, 2k+1) / [e^(w, h, 2k) + e^(w, h, 2k+1)]
            '''
            p_k = ord_labels[:, k, :, :]
            p_k = p_k.view(N, 1, H, W)

            mask_0 = (target <= k).detach()
            mask_1 = (target > k).detach()
            # print('p_k size:', p_k.size())
            # print('mask 0 size:', mask_0.size())
            # print('mask 1 size:', mask_1.size())
            '''
            对每个像素而言，
            如果k小于l(w, h), log(p_k)
            如果k大于l(w, h), log(1-p_k)
            '''

            one = torch.ones(p_k[mask_1].size())
            if torch.cuda.is_available():
                one = one.cuda()
            self.loss += torch.sum(torch.log(p_k[mask_0])) + torch.sum(torch.log(one - p_k[mask_1]))

        N = N * H * W
        self.loss /= N
        return self.loss
