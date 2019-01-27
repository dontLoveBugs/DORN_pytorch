# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/24 0:30
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import numpy as np
import os
import cv2
from tqdm import tqdm
import sys

sys.path.append('../utils')
from tool.utils import *

data_path = '/home/data/UnsupervisedDepth/KITTI-raw/kitti_raw_data/'
train_depth_dir = '/home/data/UnsupervisedDepth/KITTI-raw/train_gt16bit/'
test_depth_dir = '/home/data/UnsupervisedDepth/KITTI-raw/test_gt16bit/'
val_depth_dir = '/home/data/UnsupervisedDepth/KITTI-raw/val_gt16bit/'

if not os.path.isdir(train_depth_dir):
    os.makedirs(train_depth_dir)

if not os.path.isdir(test_depth_dir):
    os.makedirs(test_depth_dir)

if not os.path.isdir(val_depth_dir):
    os.makedirs(val_depth_dir)

max_depth = 80
img_width = 621
img_height = 188

img_files = []
img_labels = []


train_fid = open('./filenames/eigen_train_files.txt', 'r')
test_fid = open('./filenames/eigen_test_files.txt', 'r')
val_fid = open('./filenames/eigen_val_files.txt')

train_lines = train_fid.readlines()
test_lines = test_fid.readlines()
val_lines = val_fid.readlines()


print('train images num is ', len(train_lines))
print('test images num is ', len(test_lines))
print('val images num is ', len(val_lines))

train_rgbd_fid = open('eigen_train_pairs.txt', 'w')
test_rgbd_fid = open('eigen_test_pairs.txt', 'w')
val_rgbd_fid = open('eigen_val_pairs.txt', 'w')


print('Processing training images')
for in_idx in tqdm(range(len(train_lines))):
    # print(train_lines[in_idx])

    img_lines0 = train_lines[in_idx].split(' ')[0]
    index_f = str(in_idx + 1)
    img_name = index_f.zfill(5) + '.png'

    # load image and depth
    gt_file, gt_calib, im_size, im_file, cams = read_file_data_new(train_lines[in_idx], data_path)
    # print('cams:', cams)
    camera_id = cams[0]
    depth = generate_depth_map(gt_calib[0], gt_file[0], im_size[0], camera_id, False, True)
    print(depth)
    print(np.max(depth), np.min(depth))

    im_depth_16 = (depth * 100).astype(np.uint16)
    print(np.max(im_depth_16), np.min(im_depth_16))

    filename2 = os.path.join(train_depth_dir, img_name)
    file_line = os.path.join('kitti_raw_data', img_lines0) + ' ' + os.path.join('train_gt16bit', img_name) + '\n'
    train_rgbd_fid.write(file_line)
    cv2.imwrite(filename2, im_depth_16)
    exit(-1)


print('Processing test images')
for in_idx in tqdm(range(len(test_lines))):
    img_lines0 = test_lines[in_idx].split(' ')[0]
    index_f = str(in_idx + 1)
    img_name = index_f.zfill(5) + '.png'

    # load image and depth
    gt_file, gt_calib, im_size, im_file, cams = read_file_data_new(test_lines[in_idx], data_path)
    camera_id = cams[0]

    depth = generate_depth_map(gt_calib[0], gt_file[0], im_size[0], camera_id, False, True)
    im_depth_16 = (depth * 100).astype(np.uint16)

    filename2 = os.path.join(test_depth_dir, img_name)
    file_line = os.path.join('kitti_raw_data', img_lines0) + ' ' + os.path.join('test_gt16bit', img_name) + '\n'
    test_rgbd_fid.write(file_line)
    cv2.imwrite(filename2, im_depth_16)

print('Processing val images')
for in_idx in tqdm(range(len(val_lines))):
    img_lines0 = val_lines[in_idx].split(' ')[0]
    index_f = str(in_idx + 1)
    img_name = index_f.zfill(5) + '.png'

    # load image and depth
    gt_file, gt_calib, im_size, im_file, cams = read_file_data_new(val_lines[in_idx], data_path)
    camera_id = cams[0]

    depth = generate_depth_map(gt_calib[0], gt_file[0], im_size[0], camera_id, False, True)
    im_depth_16 = (depth * 100).astype(np.uint16)

    filename2 = os.path.join(val_depth_dir, img_name)
    file_line = os.path.join('kitti_raw_data', img_lines0) + ' ' + os.path.join('val_gt16bit', img_name) + '\n'
    val_rgbd_fid.write(file_line)
    cv2.imwrite(filename2, im_depth_16)


train_fid.close()
train_rgbd_fid.close()
test_fid.close()
test_rgbd_fid.close()
val_fid.close()
val_rgbd_fid.close()
