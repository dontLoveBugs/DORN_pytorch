#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-04-02 00:23
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : __init__.py.py
"""


def _get_model(cfg):
    mod = __import__('{}.{}'.format(__name__, cfg['model']['name']), fromlist=[''])
    return getattr(mod, "DepthPredModel")(**cfg["model"]["params"])