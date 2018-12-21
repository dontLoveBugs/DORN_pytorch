# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/25 14:47
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""
import torch

alpha = torch.tensor(1.0)
beta = torch.tensor(10.0)
K = 80


def get_sid_k(t):
    k = K * torch.log(t / alpha) / torch.log(beta / alpha)
    return k.int()


t = torch.ones(2, 1, 3, 3) * 5 + torch.randn(2, 1, 3, 3)
print(t)

print(get_sid_k(t))
