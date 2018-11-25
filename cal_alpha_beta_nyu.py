# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/25 11:15
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""
import os

import torch

import numpy as np

from dataloaders import nyu_dataloader

alpha = np.inf
beta = 0


def NYUDepth_loader(data_path, batch_size=32, isTrain=True):
    if isTrain:
        traindir = os.path.join(data_path, 'train')
        print(traindir)

        if os.path.exists(traindir):
            print('训练集目录存在')
        trainset = nyu_dataloader.NYUDataset(traindir, type='train')
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True)  # @wx 多线程读取失败
        return train_loader
    else:
        valdir = os.path.join(data_path, 'val')
        print(valdir)

        if os.path.exists(valdir):
            print('测试集目录存在')
        valset = nyu_dataloader.NYUDataset(valdir, type='val')
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=1, shuffle=False  # shuffle 测试时是否设置成False batch_size 恒定为1
        )
        return val_loader

data_path = '/home/data/model/wangxin/nyudepthv2'
batch_size = 16

train_loader = NYUDepth_loader(data_path, batch_size, isTrain=True)
val_loader = NYUDepth_loader(data_path, batch_size, isTrain=False)

for i, (input, target) in enumerate(train_loader):
    input, target = input.cuda(), target.cuda()
    print('train ', i)
    valid_mask = (target > 0).detach()
    max = torch.max(target[valid_mask])
    min = torch.min(target[valid_mask])

    if alpha > min:
        alpha = min

    if beta < max:
        beta = max

for i, (input, target) in enumerate(val_loader):
    input, target = input.cuda(), target.cuda()
    print('val ', i)
    valid_mask = (target > 0).detach()
    max = torch.max(target[valid_mask])
    min = torch.min(target[valid_mask])

    if alpha > min:
        alpha = min

    if beta < max:
        beta = max

print(alpha, beta)



