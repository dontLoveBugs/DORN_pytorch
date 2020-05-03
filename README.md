# DORN
### Update
The entire codebase has been updated, and some layers and loss functions have been reimplemented to make it running fast and using less memory. This respository only contains the core code of DORN model. The whole code will be saved in [SupervisedDepthPrediction](https://github.com/dontLoveBugs/SupervisedDepthPrediction).


### Introduction
This is a PyTorch implementation of [Deep Ordinal Regression Network for Monocular Depth Estimation](http://arxiv.org/abs/1806.02446). 

### Datasets

#### NYU Depth V2
Not Implemented.
 
#### KITTI
According to [the pull request](https://github.com/dontLoveBugs/DORN_pytorch/pull/19), we should move away from eigen split and switch to [kiiti depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion). More details, please see [SupervisedDepthPrediction](https://github.com/dontLoveBugs/SupervisedDepthPrediction).

