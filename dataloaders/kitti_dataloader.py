# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/23 23:00
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""


import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path, rgb=True):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if rgb:
            return img.convert('RGB')
        else:
            return img.convert('I')


def readPathFiles(file_path, root_dir):
    im_gt_paths = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            im_path = os.path.join(root_dir, line.split()[0])
            gt_path = os.path.join(root_dir, line.split()[1])

            im_gt_paths.append((im_path, gt_path))

    return im_gt_paths


# array to tensor
from dataloaders import transforms as my_transforms
to_tensor = my_transforms.ToTensor()


class KittiFolder(Dataset):
    """
        RGB:
        kitti_raw_data/2011-xx-xx/2011_xx_xx_drive_xxxx_sync/image_02/data/xxxxxxxx01.png
        Depth:
        train: train_gt16bit/xxxxx.png
        val: val_gt16bit/xxxxx.png
        test: test_gt16bit/xxxxx.png
    """

    def __init__(self, root_dir='/home/data/UnsupervisedDepth/wangixn/KITTI',
                 mode='train', loader=pil_loader, size=(385, 513)):
        super(KittiFolder, self).__init__()
        self.root_dir = root_dir

        self.mode = mode
        self.im_gt_paths = None
        self.loader = loader
        self.size = size

        if self.mode == 'train':
            self.im_gt_paths = readPathFiles('../tool/filenames/eigen_train_pairs.txt', root_dir)

        elif self.mode == 'test':
            self.im_gt_paths = readPathFiles('../tool/filenames/eigen_test_pairs.txt', root_dir)

        elif self.mode == 'val':
            self.im_gt_paths = readPathFiles('../tool/filenames/eigen_val_pairs.txt', root_dir)

        else:
            print('no mode named as ', mode)
            exit(-1)

    def __len__(self):
        return len(self.im_gt_paths)

    def train_transform(self, im, gt):
        im = np.array(im).astype(np.float32)
        gt = np.array(gt).astype(np.float32)

        s = np.random.uniform(1.0, 1.5)  # random scaling
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
        color_jitter = my_transforms.ColorJitter(0.4, 0.4, 0.4)

        transform = my_transforms.Compose([
            my_transforms.Crop(130, 10, 240, 1200),
            my_transforms.Rotate(angle),
            my_transforms.Resize(s),
            my_transforms.CenterCrop(self.size),
            my_transforms.HorizontalFlip(do_flip)
        ])

        im_ = transform(im)
        im_ = color_jitter(im_)

        gt_ = transform(gt)

        im_ = np.array(im_).astype(np.float32)
        gt_ = np.array(gt_).astype(np.float32)

        im_ /= 255.0
        gt_ /= 100.0 * s
        im_ = to_tensor(im_)
        gt_ = to_tensor(gt_)

        gt_ = gt_.unsqueeze(0)

        return im_, gt_

    def val_transform(self, im, gt):
        im = np.array(im).astype(np.float32)
        gt = np.array(gt).astype(np.float32)

        transform = my_transforms.Compose([
            my_transforms.Crop(130, 10, 240, 1200),
            my_transforms.CenterCrop(self.size)
        ])

        im_ = transform(im)
        gt_ = transform(gt)

        im_ = np.array(im_).astype(np.float32)
        gt_ = np.array(gt_).astype(np.float32)

        im_ /= 255.0
        gt_ /= 100.0
        im_ = to_tensor(im_)
        gt_ = to_tensor(gt_)

        gt_ = gt_.unsqueeze(0)
        return im_, gt_

    def __getitem__(self, idx):
        im_path, gt_path = self.im_gt_paths[idx]

        if self.mode == 'train':
            im_path = os.path.join(self.root_dir, 'kitti_raw_data', im_path)

        im = self.loader(im_path)
        gt = self.loader(gt_path, rgb=False)

        if self.mode == 'train':
            im, gt = self.train_transform(im, gt)

        else:
            im, gt = self.val_transform(im, gt)

        return im, gt


import torch

if __name__ == '__main__':
    root_dir = '/home/data/UnsupervisedDepth/wangixn/KITTI'

    # im_gt_paths = readPathFiles('./eigen_val_pairs.txt', root_dir)

    data_set = KittiFolder(root_dir, mode='train', size=(228, 912))
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=0)

    print('dataset num is ', len(data_loader))

    for i, (im, gt) in enumerate(data_loader):

        # print(im)

        valid = (gt > 0.0)
        print(torch.max(gt[valid]), torch.min(gt[valid]))
        # print(gt.size())
        print(im.size())
        # print('im size:', im.size())
        # print('gt size:', gt.size())

        # print(gt)
        # print(torch.max(gt))
        # print(torch.min(gt))

        # print(im)
        #
        # if i == 0:
        #     break
