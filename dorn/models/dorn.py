#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 21:06
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : dorn.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dorn.modules.backbones.resnet import ResNetBackbone
from dorn.modules.encoders.KITTISceneModule import SceneUnderstandingModule as KittiSceneModule
from dorn.modules.encoders.NYUSceneModule import SceneUnderstandingModule as NyuSceneModule
from dorn.modules.decoders.OrdinalRegression import OrdinalRegressionLayer
from dorn.modules.losses.ordinal_regression_loss import OrdinalRegressionLoss


class DepthPredModel(nn.Module):

    def __init__(self, ord_num=90, gamma=1.0, beta=80.0, scene="Kitti", discretization="SID", pretrained=True):
        super().__init__()
        self.ord_num = ord_num
        self.gamma = gamma
        self.beta = beta
        self.discretization = discretization

        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.SceneUnderstandingModule = KittiSceneModule(ord_num) if scene == "Kitti" else NyuSceneModule(ord_num)
        self.regression_layer = OrdinalRegressionLayer()
        self.criterion = OrdinalRegressionLoss(ord_num, beta, discretization)

    def forward(self, image, target=None):
        N, C, H, W = image.shape
        feat = self.backbone(image)
        feat = self.SceneUnderstandingModule(feat)
        if self.training:
            prob = self.regression_layer(feat)
            loss = self.criterion(prob, target)
            return loss

        prob, label = self.regression_layer(feat)
        # print("prob shape:", prob.shape, " label shape:", label.shape)
        if self.discretization == "SID":
            t0 = torch.exp(np.log(self.beta) * label.float() / self.ord_num)
            t1 = torch.exp(np.log(self.beta) * (label.float() + 1) / self.ord_num)
        else:
            t0 = 1.0 + (self.beta - 1.0) * label.float() / self.ord_num
            t1 = 1.0 + (self.beta - 1.0) * (label.float() + 1) / self.ord_num
        depth = (t0 + t1) / 2 - self.gamma
        # print("depth min:", torch.min(depth), " max:", torch.max(depth),
        #       " label min:", torch.min(label), " max:", torch.max(label))
        return {"target": [depth], "prob": [prob], "label": [label]}
