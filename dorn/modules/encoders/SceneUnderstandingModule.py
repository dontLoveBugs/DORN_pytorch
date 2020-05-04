#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-04 15:17
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : SceneUnderstandingModule.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dorn.modules.layers.basic_layers import conv_bn_relu


class FullImageEncoder(nn.Module):
    def __init__(self, h, w, kernel_size):
        super(FullImageEncoder, self).__init__()
        self.global_pooling = nn.AvgPool2d(kernel_size, stride=kernel_size, padding=kernel_size // 2)  # KITTI 16 16
        self.dropout = nn.Dropout2d(p=0.5)
        self.h = h // kernel_size + 1
        self.w = w // kernel_size + 1
        # print("h=", self.h, " w=", self.w, h, w)
        self.global_fc = nn.Linear(2048 * self.h * self.w, 512)  # kitti 4x5
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # 1x1 卷积

    def forward(self, x):
        # print('x size:', x.size())
        x1 = self.global_pooling(x)
        # print('# x1 size:', x1.size())
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048 * self.h * self.w)  # kitti 4x5
        x4 = self.relu(self.global_fc(x3))
        # print('# x4 size:', x4.size())
        x4 = x4.view(-1, 512, 1, 1)
        # print('# x4 size:', x4.size())
        x5 = self.conv1(x4)
        # out = self.upsample(x5)
        return x5


class SceneUnderstandingModule(nn.Module):
    def __init__(self, ord_num, size, kernel_size, pyramid=[6, 12, 18], batch_norm=False):
        # pyramid kitti [6, 12, 18] nyu [4, 8, 12]
        super(SceneUnderstandingModule, self).__init__()
        assert len(size) == 2
        assert len(pyramid) == 3
        self.size = size
        h, w = self.size
        self.encoder = FullImageEncoder(h // 8, w // 8, kernel_size)
        self.aspp1 = nn.Sequential(
            conv_bn_relu(batch_norm, 2048, 512, kernel_size=1, padding=0),
            conv_bn_relu(batch_norm, 512, 512, kernel_size=1, padding=0)
        )
        self.aspp2 = nn.Sequential(
            conv_bn_relu(batch_norm, 2048, 512, kernel_size=3, padding=pyramid[0], dilation=pyramid[0]),
            conv_bn_relu(batch_norm, 512, 512, kernel_size=1, padding=0)
        )
        self.aspp3 = nn.Sequential(
            conv_bn_relu(batch_norm, 2048, 512, kernel_size=3, padding=pyramid[1], dilation=pyramid[1]),
            conv_bn_relu(batch_norm, 512, 512, kernel_size=1, padding=0)
        )
        self.aspp4 = nn.Sequential(
            conv_bn_relu(batch_norm, 2048, 512, kernel_size=3, padding=pyramid[2], dilation=pyramid[2]),
            conv_bn_relu(batch_norm, 512, 512, kernel_size=1, padding=0)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            conv_bn_relu(batch_norm, 512*5, 2048, kernel_size=1, padding=0),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, ord_num * 2, 1)
        )

    def forward(self, x):
        N, C, H, W = x.shape
        x1 = self.encoder(x)
        x1 = F.interpolate(x1, size=(H, W), mode="bilinear", align_corners=True)

        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)

        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.concat_process(x6)
        out = F.interpolate(out, size=self.size, mode="bilinear", align_corners=True)
        return out
