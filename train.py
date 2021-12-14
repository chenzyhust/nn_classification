# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from icecream import ic
from config.config import get_cfg_defaults
from nncls.utils import *
from nncls.datasets import (build_train_loader, build_test_loader, 
                           build_transforms, aug_data)
from nncls.models import build_model
from nncls.losses import build_train_loss, build_aug_loss
from nncls.optim import build_optim
from nncls.scheduler import build_scheduler, WarmUpLR

try:
    import apex
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

def swa_schedule(epoch):
    t = (epoch) / (cfg.train.swa_start)
    lr_ratio = cfg.train.swa_lr / cfg.train.lr
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return cfg.train.lr * factor

def train(epoch):
    start = time.time()
    model.train()
    for batch_index, (images, labels) in enumerate(train_loader):
        labels = labels.cuda()
        images = images.cuda()
        r = np.random.rand(1)
        #aug_images, aug_labels = images, labels
        if cfg.train.optim == 'sam':
            def closure():
                optimizer.zero_grad()
                if r < cfg.aug.prob:
                    aug_images, aug_labels = aug_data(cfg, images, labels)
                    outputs, features = model(aug_images)
                    loss = aug_loss(outputs, aug_labels)
                else:
                    aug_images, aug_labels = images, labels
                    outputs, features = model(aug_images)
                    loss = train_loss(outputs, aug_labels)
                if cfg.apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                return loss
            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            if r < cfg.aug.prob:
                aug_images, aug_labels = aug_data(cfg, images, labels)
                outputs, features = model(aug_images)
                loss = aug_loss(outputs, aug_labels)
            else:
                aug_images, aug_labels = images, labels
                outputs, features = model(aug_images)
                loss = train_loss(outputs, aug_labels)
            
            # outputs = net(aug_images)
            # loss = train_loss(outputs, aug_labels)
            if cfg.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            optimizer.step()
        if epoch <= cfg.train.warm:
            warmup_scheduler.step()
        
        loss_all_reduce = dist.all_reduce(loss,
                                          op=dist.ReduceOp.SUM,
                                          async_op=True)
        loss_all_reduce.wait()
        loss.div_(dist.get_world_size())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if dist.get_rank() == 0:
            print('training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * cfg.train.batch * dist.get_world_size() + len(images) * dist.get_world_size(),
                total_samples=len(train_loader.dataset)
            ))
    if dist.get_rank() == 0:
        finish = time.time()
        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
   
@torch.no_grad()
def eval_training(model, epoch=0, suffix=''):

    start = time.time()
    model.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    for (images, labels) in test_loader:

        images = images.cuda()
        labels = labels.cuda()

        outputs, features = model(images)
        loss = val_loss(outputs, labels)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 1))
       
        loss_all_reduce = dist.all_reduce(loss,
                                          op=dist.ReduceOp.SUM,
                                          async_op=True)
        acc1_all_reduce = dist.all_reduce(acc1,
                                          op=dist.ReduceOp.SUM,
                                          async_op=True)
        acc5_all_reduce = dist.all_reduce(acc5,
                                          op=dist.ReduceOp.SUM,
                                          async_op=True)
        loss_all_reduce.wait()
        acc1_all_reduce.wait()
        acc5_all_reduce.wait()
        loss.div_(dist.get_world_size())
        acc1.div_(dist.get_world_size())
        acc5.div_(dist.get_world_size())
        loss = loss.item()
        acc1 = acc1.item()
        acc5 = acc5.item()
        num = images.size(0)
        loss_meter.update(loss, num)
        acc1_meter.update(acc1, num)
        acc5_meter.update(acc5, num)
        if torch.cuda.is_available():
            torch.cuda.synchronize()     
    if dist.get_rank() == 0:
        finish = time.time()
        if suffix == '':
            # print('GPU INFO.....')
            # print(torch.cuda.memory_summary(), end='')
            print('Evaluating Network.....')
        print('{} Test set: Epoch: {}, Avg loss: {:.4f}, Top 1 Acc: {:.4f}, Top 5 Acc: {:.4f}, Time consumed:{:.2f}s'.format(
            suffix,
            epoch,
            loss_meter.avg,
            acc1_meter.avg,
            acc5_meter.avg,
            finish - start
        ))
        print()
        #add informations to tensorboard
    
    return acc1_meter.avg, acc5_meter.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.local_rank = args.local_rank
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.train.gpu_id

    torch.backends.cudnn.deterministic = cfg.train.detem 
    torch.backends.cudnn.benchmark = cfg.train.cudnn
    torch.cuda.set_device(cfg.local_rank)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    set_seed(1 + rank)
    if dist.get_rank() == 0:
        ic(cfg.train.gpu_id)
    
    model = build_model(cfg)
    if cfg.apex and cfg.sync_bn:
        model = apex.parallel.convert_syncbn_model(model)
        if dist.get_rank() == 0:
            print('using apex synced BN')
    # data loader:
    train_loader = build_train_loader(cfg)
    test_loader = build_test_loader(cfg)
    # loss function:
    train_loss = build_train_loss(cfg)
    aug_loss = build_aug_loss(cfg)
    val_loss = train_loss
    # optimizer:
    optimizer = build_optim(cfg, model)
    # swa setting:
    swa_model = None
    if cfg.train.scheduler == 'step' and cfg.train.swa:
        swa_model = deepcopy(model)
        swa_n = 0
    # 通过调整下面的opt_level实现半精度训练。
    # opt_level选项有：'O0', 'O1', 'O2', 'O3'.
    # 其中'O0'是fp32常规训练，'O1'、'O2'是fp16训练，'O3'则可以用来推断但不适合拿来训练（不稳定）
    # 注意，当选用fp16模式进行训练时，keep_batchnorm默认是None，无需设置；
    # scale_loss是动态模式，可以设置也可以不设置。
    if cfg.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=cfg.opt_level)
        model = ApexDDP(model)
    # scheduler:
    train_scheduler = build_scheduler(cfg, optimizer)
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * cfg.train.warm)
    resume_epoch = 1
    # setup exponential moving average of model weights, SWA could be used here too
        
    if dist.get_rank() == 0:
        checkpoint_path = os.path.join(cfg.workspace, cfg.net, cfg.time_now)

        #create checkpoint folder to save model
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

        best_acc1 = 0.0
        bets_acc5 = 0.0
        best_epoch = 0
        if cfg.train.swa:
            best_swa_acc1 = 0.0
            best_swa_acc5 = 0.0
        if cfg.train.resume:
            weights_path = cfg.train.resume_path
            model.module.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))
            resume_epoch = 1
    
    save_epoch_thr = int(0.5 * cfg.train.epoches)
    for epoch in range(1, cfg.train.epoches):
        if epoch > cfg.train.warm:
            if cfg.train.swa:
                lr = swa_schedule(epoch)
                adjust_learning_rate(optimizer, lr)
            else:
                train_scheduler.step(epoch)

        if cfg.train.resume:
            if epoch <= resume_epoch:
                continue
        train_loader.sampler.set_epoch(epoch)
        train(epoch)
        acc1, acc5 = eval_training(model, epoch)
        if cfg.train.swa and  epoch >= cfg.train.swa_start:
            moving_average(swa_model, model, 1.0 / (swa_n + 1))
            swa_n += 1
            bn_update(train_loader, swa_model)
            swa_acc1, swa_acc5 = eval_training(swa_model, epoch, suffix='SWA')
            if dist.get_rank() == 0:
                if best_swa_acc1 < swa_acc1:
                    best_swa_acc1 = swa_acc1
                    best_swa_acc5 = swa_acc5
                print('best swa acc1: {:.4f}, best swa acc5: {:.4f}'.format(best_swa_acc1, best_swa_acc5))
                print()
        #start to save best performance model after learning rate decay to 0.01
        if dist.get_rank() == 0:
            if epoch > save_epoch_thr and best_acc1 < acc1:
                prefix = os.path.join(cfg.workspace, cfg.net)
                savemodel(cfg, model, prefix, epoch)
                best_acc1 = acc1
                best_acc5 = acc5
                if best_epoch > 0:
                    os.remove(prefix + "_epoch_" + str(best_epoch) + '.pth')
                    os.remove(prefix + "_epoch_" + str(best_epoch) + '.onnx')
                    os.remove(prefix + "_epoch_" + str(best_epoch) + '_simplified.onnx')
                best_epoch = epoch
                print('best epoch: {}, best acc1: {:.4f}, acc5: {:.4f}'.format(best_epoch, best_acc1, best_acc5))
                print()
                continue
            if epoch > save_epoch_thr:
                print('best epoch: {}, best acc1: {:.4f}, acc5: {:.4f}'.format(best_epoch, best_acc1, best_acc5))
                print()
                
