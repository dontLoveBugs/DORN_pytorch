# DORN implemented in Pytorch 0.4.1


### Introduction
This is a PyTorch(0.4.1) implementation of [Deep Ordinal Regression Network for Monocular Depth Estimation](http://arxiv.org/abs/1806.02446). At present, we can provide train script in NYU Depth V2 dataset. KITTI will be available soon!

Note: we modify the ordinal layer using matrix operation, making trianing faster.

### TODO
- [x] DORN model in nyu and kitti
- [x] Training DORN on nyu and kitti datasets
- [ ] Results evaluation on nyu test set
- [x] Calculate alpha and beta in nyu dataset
- [x] Realize the ordinal loss in paper 
