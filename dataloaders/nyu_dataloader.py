import torch

import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

iheight, iwidth = 480, 640  # raw image size
alpha, beta = 0.02, 10.0  # NYU Depth, min depth is 0.02m, max depth is 10.0m
K = 68  # NYU is 68, but in paper, 80 is good

'''
In this paper, all the images are reduced to 288 x 384 from 480 x 640,
And the model are trained on random crops of size 257x353.
'''


class NYUDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(NYUDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (257, 353)

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(288.0 / iheight),  # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np)  # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, self.get_depth_sid(depth_np)

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(288.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def get_depth_sid(self, depth):
        k = K * np.log(depth / alpha) / np.log(beta / alpha)
        k = k.astype(np.int32)
        return k


"""
After obtaining ordinal labels for each position od Image, 
the predicted depth value d(w, h) can be decoded as below.
"""


def get_depth_sid(depth_labels):
    if torch.cuda.is_available():
        alpha_ = torch.tensor(0.02).cuda()
        beta_ = torch.tensor(10.0).cuda()
        K_ = torch.tensor(68.0).cuda()
    else:
        alpha_ = torch.tensor(0.02)
        beta_ = torch.tensor(10.0)
        K_ = torch.tensor(68.0)

    t = torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * depth_labels / K_)
    depth = t
    return depth
