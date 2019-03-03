# DORN implemented in Pytorch 0.4.1


### Introduction
This is a PyTorch(0.4.1) implementation of [Deep Ordinal Regression Network for Monocular Depth Estimation](http://arxiv.org/abs/1806.02446). At present, we can provide train script in NYU Depth V2 dataset and Kitti Dataset!

Note: we modify the ordinal layer using matrix operation, making trianing faster.

### TODO
- [x] DORN model in nyu and kitti
- [x] Training DORN on nyu and kitti datasets
- [ ] Results evaluation on nyu test set
- [x] the script to generate nyu and kitti dataset.
- [x] Calculate alpha and beta in nyu dataset and kitti dataset
- [x] Realize the ordinal loss in paper 

### Datasets

#### NYU Depth V2
DORN need to use all the Images (about 120k) in the dataset, but if you just want to test the code, you can use the nyu_depth_v2_labeled.mat and turn it to a h5 file. The convert script is 'create_nyu_h5.py' and you need to change the file paths to yours.

 - Modify create_nyu_h5.py with your path and run the script.
 
  ```bash
  python create_nyu_h5.py
  ```
 
#### Kitti
The kitti dataset contains 23488 images from 32 scenes for training and 697 images from the remaining 29 scenes for testing. 
 - Raw dataset (about 175 GB) can be downloaded by running:
  ```bash
  wget -i kitti_archives_to_download.txt -P ~/kitti-raw-data/
  ```
 - Unzip the compressed files:
  ```bash
  cd ~/kitti-raw-data
  find . -name '*.zip' -exec unzip {} \;
  ```
 - Run the script to generate the kitti_ground_truth
  ```
  python gen_kitti_dataset.py
  ```

