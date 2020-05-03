# DORN
### Update
The entire codebase has been updated, and some layers and loss functions have been reimplemented to make it running fast and using less memory. This respository only contains the core code of DORN model. The whole code will be saved in [SupervisedDepthPrediction](https://github.com/dontLoveBugs/SupervisedDepthPrediction).


### Introduction
This is a PyTorch implementation of [Deep Ordinal Regression Network for Monocular Depth Estimation](http://arxiv.org/abs/1806.02446). 

### Pretrained Model
The resnet backbone of DORN, which has three conv in first conv layer, is different from original resnet. The pretrained model of the resnet backbone can download from [MIT imagenet pretrained resnet101](http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth).

### Datasets

#### NYU Depth V2
Not Implemented.
 
#### KITTI
According to [the pull request](https://github.com/dontLoveBugs/DORN_pytorch/pull/19), we should move away from eigen split and switch to [kiiti depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion). More details, please see [SupervisedDepthPrediction](https://github.com/dontLoveBugs/SupervisedDepthPrediction).

