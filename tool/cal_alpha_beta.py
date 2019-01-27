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
from dataloaders.kitti_dataloader import KittiFolder
from dataloaders.path import Path


def create_loader(dataset='kitti'):
    root_dir = Path.db_root_dir(dataset)
    if dataset == 'kitti':
        train_set = KittiFolder(root_dir, mode='train', size=(228, 912))
        test_set = KittiFolder(root_dir, mode='test', size=(228, 912))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=False,
                                                   num_workers=0, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False,
                                                  num_workers=0, pin_memory=True)
        return train_loader, test_loader
    else:
        traindir = os.path.join(root_dir, 'train')
        if os.path.exists(traindir):
            print('Train dataset "{}" is existed!'.format(traindir))
        else:
            print('Train dataset "{}" is not existed!'.format(traindir))
            exit(-1)

        valdir = os.path.join(root_dir, 'val')
        if os.path.exists(traindir):
            print('Train dataset "{}" is existed!'.format(valdir))
        else:
            print('Train dataset "{}" is not existed!'.format(valdir))
            exit(-1)

        train_set = nyu_dataloader.NYUDataset(traindir, type='train')
        val_set = nyu_dataloader.NYUDataset(valdir, type='val')

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

        return train_loader, val_loader


def cal_alpga_beta(args):
    train_loader, val_loader = create_loader(args.dataset)

    alpha = np.inf
    beta = 0

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

    return alpha, beta


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='cal alpha and beta')
    parser.add_argument('--dataset', default='kitti', type=str)

    args = parser.parse_args()
    cal_alpga_beta(args)
