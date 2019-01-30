# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/21 15:25
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

from datetime import datetime
import shutil
import socket
import time
import torch
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

from dataloaders import nyu_dataloader
from dataloaders.kitti_dataloader import KittiFolder
from dataloaders.path import Path
from metrics import AverageMeter, Result
import utils
import criteria
import os
import torch.nn as nn

import numpy as np
import random

from network.get_models import get_models

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use single GPU

args = utils.parse_command()
print(args)

# if setting gpu id, the using single GPU
if args.gpu:
    print('Single GPU Mode.')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

best_result = Result()
best_result.set_to_worst()


def create_loader(args):
    root_dir = Path.db_root_dir(args.dataset)
    if args.dataset == 'kitti':
        train_set = KittiFolder(root_dir, mode='train', size=(385, 513))
        test_set = KittiFolder(root_dir, mode='test', size=(385, 513))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True)
        return train_loader, test_loader
    else:
        traindir = os.path.join(root_dir, 'train')
        if os.path.exists(traindir):
            print('Train dataset "{}" is existed!'.format(traindir))
        else:
            print('Train dataset "{}" is not existed!'.format(traindir))
            exit(-1)

        valdir = os.path.join(root_dir, 'val')
        if os.path.exists(traindir):
            print('Train dataset "{}" is existed!'.format(valdir))
        else:
            print('Train dataset "{}" is not existed!'.format(valdir))
            exit(-1)

        train_set = nyu_dataloader.NYUDataset(traindir, type='train')
        val_set = nyu_dataloader.NYUDataset(valdir, type='val')

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

        return train_loader, val_loader


def main():
    global args, best_result, output_directory

    # set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        args.batch_size = args.batch_size * torch.cuda.device_count()
    else:
        print("Let's use GPU ", torch.cuda.current_device())

    train_loader, val_loader = create_loader(args)

    if args.resume:
        assert os.path.isfile(args.resume), \
            "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)

        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        optimizer = checkpoint['optimizer']

        # solve 'out of memory'
        model = checkpoint['model']

        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

        # clear memory
        del checkpoint
        # del model_dict
        torch.cuda.empty_cache()
    else:
        print("=> creating Model")
        model = get_models(args.dataset)
        print("=> model created.")
        start_epoch = 0

        # different modules have different learning rate
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # You can use DataParallel() whether you use Multi-GPUs or not
        model = nn.DataParallel(model).cuda()

    # when training, use reduceLROnPlateau to reduce learning rate
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.lr_patience)

    # loss function
    criterion = criteria.ordLoss()

    # create directory path
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    best_txt = os.path.join(output_directory, 'best.txt')
    config_txt = os.path.join(output_directory, 'config.txt')

    # write training parameters to config file
    if not os.path.exists(config_txt):
        with open(config_txt, 'w') as txtfile:
            args_ = vars(args)
            args_str = ''
            for k, v in args_.items():
                args_str = args_str + str(k) + ':' + str(v) + ',\t\n'
            txtfile.write(args_str)

    # create log
    log_path = os.path.join(output_directory, 'logs',
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    for epoch in range(start_epoch, args.epochs):

        # remember change of the learning rate
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = float(param_group['lr'])
            logger.add_scalar('Lr/lr_' + str(i), old_lr, epoch)

        train(train_loader, model, criterion, optimizer, epoch, logger)  # train for one epoch
        result, img_merge = validate(val_loader, model, epoch, logger)  # evaluate on validation set

        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write(
                    "epoch={}, rmse={:.3f}, rml={:.3f}, log10={:.3f}, d1={:.3f}, d2={:.3f}, dd31={:.3f}, "
                    "t_gpu={:.4f}".
                        format(epoch, result.rmse, result.absrel, result.lg10, result.delta1, result.delta2,
                               result.delta3,
                               result.gpu_time))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)

        # save checkpoint for each epoch
        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'model': model,
            'best_result': best_result,
            'optimizer': optimizer,
        }, is_best, epoch, output_directory)

        # when rml doesn't fall, reduce learning rate
        scheduler.step(result.absrel)

    logger.close()


# train
def train(train_loader, model, criterion, optimizer, epoch, logger):
    average_meter = AverageMeter()
    model.train()  # switch to train mode
    end = time.time()

    batch_num = len(train_loader)

    for i, (input, target) in enumerate(train_loader):

        # itr_count += 1
        input, target = input.cuda(), target.cuda()
        # print('input size  = ', input.size())
        # print('target size = ', target.size())
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()

        with torch.autograd.detect_anomaly():
            pred_d, pred_ord = model(input)  # @wx 注意输出
            target_c = utils.get_labels_sid(args, target)  # using sid, discretize the groundtruth
            loss = criterion(pred_ord, target_c)
            optimizer.zero_grad()
            loss.backward()  # compute gradient and do SGD step
            optimizer.step()

        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        depth = utils.get_depth_sid(args, pred_d)
        result.evaluate(depth.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'Loss={Loss:.5f} '
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'RML={result.absrel:.2f}({average.absrel:.2f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                epoch, i + 1, len(train_loader), data_time=data_time,
                gpu_time=gpu_time, Loss=loss.item(), result=result, average=average_meter.average()))
            current_step = epoch * batch_num + i
            logger.add_scalar('Train/RMSE', result.rmse, current_step)
            logger.add_scalar('Train/rml', result.absrel, current_step)
            logger.add_scalar('Train/Log10', result.lg10, current_step)
            logger.add_scalar('Train/Delta1', result.delta1, current_step)
            logger.add_scalar('Train/Delta2', result.delta2, current_step)
            logger.add_scalar('Train/Delta3', result.delta3, current_step)

    avg = average_meter.average()


# validation
def validate(val_loader, model, epoch, logger):
    average_meter = AverageMeter()

    model.eval()  # switch to evaluate mode

    end = time.time()

    skip = len(val_loader) // 9  # save images every skip iters

    for i, (input, target) in enumerate(val_loader):

        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred, _ = model(input)

        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        pred = utils.get_depth_sid(args, pred)
        result.evaluate(pred.data, target.data)

        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        rgb = input

        if i == 0:
            img_merge = utils.merge_into_row(rgb, target, pred)
        elif (i < 8 * skip) and (i % skip == 0):
            row = utils.merge_into_row(rgb, target, pred)
            img_merge = utils.add_row(img_merge, row)
        elif i == 8 * skip:
            filename = output_directory + '/comparison_' + str(epoch) + '.png'
            utils.save_image(img_merge, filename)

        if (i + 1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'RML={result.absrel:.2f}({average.absrel:.2f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                i + 1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
          'RMSE={average.rmse:.3f}\n'
          'Rel={average.absrel:.3f}\n'
          'Log10={average.lg10:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'Delta2={average.delta2:.3f}\n'
          'Delta3={average.delta3:.3f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    logger.add_scalar('Test/rmse', avg.rmse, epoch)
    logger.add_scalar('Test/Rel', avg.absrel, epoch)
    logger.add_scalar('Test/log10', avg.lg10, epoch)
    logger.add_scalar('Test/Delta1', avg.delta1, epoch)
    logger.add_scalar('Test/Delta2', avg.delta2, epoch)
    logger.add_scalar('Test/Delta3', avg.delta3, epoch)
    return avg, img_merge


if __name__ == '__main__':
    main()
