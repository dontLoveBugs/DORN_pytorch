# -*- coding: utf-8 -*-
# @Time    : 2018/11/21 22:27
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
import os
from datetime import time

import torch

import numpy as np

import modelv1
import modelv2
import torch.nn as nn
from NYUDepth_train import NYUDepth_loader
import matplotlib.pyplot as plt
from PIL import Image

cmap = plt.cm.jet


def parse_command():
    import argparse
    parser = argparse.ArgumentParser(description='NYUDepth')
    parser.add_argument('-model_path', type=str,
                        default='/home/data/model/wangxin/Project/DepthEstimate/run/run_1/model_best.pth.tar')
    parser.add_argument('-b', '--batch-size', default=1, type=int, help='mini-batch size (default: 1)')
    parser.add_argument("--data_path", type=str, default="/home/data/model/wangxin/nyudepthv2",
                        help="the root folder of dataset")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--save_path', type=str, default="./result1/")
    args = parser.parse_args()
    return args


args = parse_command()


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = torch.sqrt(((gt - pred) ** 2).mean())
    log10_err = torch.abs(torch.log10(gt) - torch.log10(pred)).mean()
    abs_rel = (torch.abs(gt - pred) / gt).mean()

    return abs_rel, rmse, log10_err, a1, a2, a3


# 修改
def validate(val_loader, model, output_directory):
    model.eval()  # switch to evaluate mode

    rel_s = []
    rmse_s = []
    log10_s = []
    d1_s = []
    d2_s = []
    d3_s = []

    for i, (input, target) in enumerate(val_loader):

        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()

        with torch.no_grad():
            pred = model(input)
        torch.cuda.synchronize()

        rel, rmse, log10, d1, d2, d3 = compute_errors(target, pred)

        rel_s.append(rel)
        rmse_s.append(rmse)
        log10_s.append(log10)
        d1_s.append(d1)
        d2_s.append(d2)
        d3_s.append(d3)

        rel_avg = np.mean(rel_s)
        rmse_avg = np.mean(rmse_s)
        log10_avg = np.mean(log10_s)
        d1_avg = np.mean(d1_s)
        d2_avg = np.mean(d2_s)
        d3_avg = np.mean(d3_s)

        # save 8 images for visualization
        skip = 50

        rgb = input

        if i == 0:
            img_merge = merge_into_row(rgb, target, pred)
        elif (i < 8 * skip) and (i % skip == 0):
            row = merge_into_row(rgb, target, pred)
            img_merge = add_row(img_merge, row)
        elif i == 8 * skip:
            filename = output_directory + '/result'  + '.png'
            save_image(img_merge, filename)

        if (i + 1) % args.print_freq == 0:
            # print('Test: [{0}/{1}]'
            #       'rel={rel:.3f}({rel_avg:.3f})'
            #       'rmse={rmse:.3f}({rmse_avg:.3f}) '
            #       'log10={log10:.3f}({log10_qvg:.3f}) '
            #       'Delta1={d1:.3f}({d1_avg:.3f}) '
            #       'Delta2={d2:.3f}({d2_avg:.3f}) '
            #       'Delta3={d3:.3f}({d3_avg:.3f}) '.format(
            #     i + 1, len(val_loader), rel, rel_avg, rmse, rmse_avg, log10, log10_avg, d1, d1_avg, d2, d2_avg, d3, d3_avg))

            print('#', i, ':', rel.item(), rmse.item(), log10.item(), d1.item(), d2.item(), d3.item())

    return rel_avg, rmse_avg, log10_avg, d1_avg, d2_avg, d3_avg


if __name__ == "__main__":
    val_loader = NYUDepth_loader(args.data_path, batch_size=args.batch_size, isTrain=False)

    model = modelv1.ResNet(layers=50, output_size=((228, 304)))

    # load model state_dict
    checkpoint = torch.load(args.model_path)
    model_dict = checkpoint['model']
    if torch.cuda.device_count() > 1:
        model_dict = model_dict.module.state_dict()  # 如果是多卡训练的要加module
    else:
        model_dict = model.state_dict()
    model.load_state_dict(model_dict)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    print('Test result:')
    rel, rmse, log10, d1, d2, d3 = validate(val_loader, model, args.save_path)
    rel = rel.item()
    rmse = rmse.item()
    log10 = log10.item()
    d1 = d1.item()
    d2 = d2.item()
    d3 = d3.item()

    print(rel, rmse, log10, d1, d2, d3)

    best_txt = args.save_path + '/result.txt'
    with open(best_txt, 'w') as txtfile:
        txtfile.write(
            "rml={:.3f}\t\nrmse={:.3f}\t\nlog10={:.3f}\t\nd1={:.3f}\t\nd2={:.3f}\t\nd3={:.3f}\t\n".
            format(rel, rmse, log10, d1, d2, d3))