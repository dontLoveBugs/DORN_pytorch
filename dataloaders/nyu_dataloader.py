import torch

import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

iheight, iwidth = 480, 640  # raw image size
min, max = 0.0, 10.0  # NYU Depth的深度值最小为0 最大为10
alpha, beta = 0.1, 10.0
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

    # def depth_sid(self, depth, alpha, beta):
    #     internal =
    #     return

    def get_depth_sid(self, depth):
        k = K * torch.log(depth / alpha) / torch.log(beta / alpha)
        return k.int()
