# -*- coding: utf-8 -*-
# @Time    : 2018/11/21 20:46
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com


# usr/bin/bash -tt
import torch

import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
import string
from sklearn.preprocessing import normalize
from PIL import Image
from skimage.transform import resize
import cv2
import sys
import glob
import modelv2


## the function for evaluation of different metrics
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.mean((gt - pred) ** 2)

    log10_err = np.mean(np.absolute(np.log10(gt) - np.log10(pred)))

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    return abs_rel, rmse, log10_err, a1, a2, a3


reconstruction_results_dir = './results/depth';
if not os.path.isdir(reconstruction_results_dir):
    os.mkdir(reconstruction_results_dir)

raw_depth_dir = 'Path/to/rawDepth' ## the path to the raw depth data (saved in 16bit format png)

model_output_dir = '/home/data/model/wangxin/Project/DepthEstimate/run/run_4/model_best.pth.bar'  ## the path to the trained caffemodel

root_dir = './NYUD-v2';

checkpoint = torch.load(model_output_dir)

net = modelv2.ResNet()

fp = open('path/to/test_nyu.list');  ## path to the list of test files

lines = fp.readlines();
image_size = 0;
num_samples = 654
mean_values = [104.008, 116.669, 122.675]
rmse = np.zeros(num_samples, np.float32)
log10_err = np.zeros(num_samples, np.float32)
abs_rel = np.zeros(num_samples, np.float32)
a1 = np.zeros(num_samples, np.float32)
a2 = np.zeros(num_samples, np.float32)
a3 = np.zeros(num_samples, np.float32)
min_depth = 0.7
max_depth = 10
for i in range(0, len(lines)):
    print("Processing %d-th image...\n" % i)

img = cv2.imread(os.path.join(root_dir, lines[i].split(' ')[0].replace('\n', '')));
img = np.float32(img)
img = img[:, :, ::-1]
img -= mean_values;
img = img.transpose((2, 0, 1))

net.blobs['data'].data[...] = img
net.forward()
image = net.blobs['predicted-map2'].data
image = np.squeeze(image[0, :, :, :])

img_full_path = lines[i]
file_path = img_full_path.split(' ')[0];
file_name = file_path.split('/')[1]
image[image < 0.7] = 0.7
image[image > 10] = 10
raw_depth = cv2.imread(os.path.join(raw_depth_dir, file_name), -1);
raw_depth = raw_depth / float(10000)
pred_depth = cv2.resize(image, (640, 480))
mask = np.logical_and(raw_depth > min_depth, raw_depth < max_depth)
abs_rel[i], rmse[i], log10_err[i], a1[i], a2[i], a3[i] = compute_errors(raw_depth[mask], pred_depth[mask]);

# save 16 bit prediction depth
pred_16bit = (image * 10000).astype(np.uint16);

# save colormaps
max_depth = image.max()
min_depth = image.min()
image = (image - min_depth) / (max_depth - min_depth);
Image.fromarray(np.uint8(plt.cm.jet(image) * 255)).save(os.path.join(reconstruction_results_dir, file_name))

# print out average errors and accuracy
fp.close();
print("abs_rel: {:1.4f}, rms: {:1.4f}, log10: {:1.3f}, a1: {:1.3f}, a2: {:1.3f}, a3: {:1.3f}".format(abs_rel.mean(),
                                                                                                     np.sqrt(
                                                                                                         rmse.mean()),
                                                                                                     log10_err.mean(),
                                                                                                     a1.mean(),
                                                                                                     a2.mean(),
                                                                                                     a3.mean()))
print('Success!')