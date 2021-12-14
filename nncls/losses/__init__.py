#!/usr/bin/env	python3
""" build loss code
author: chenzy
date:   2021.2.18
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from icecream import ic
from .focal_loss import FocalLoss
from .label_smoothing import LabelSmoothingLoss
from .aug_loss import MixLoss, RicapLoss, DualCutoutLoss
#from .center_loss import CenterLoss

__all__ = ['build_train_loss', 'build_aug_loss']

def build_train_loss(cfg):
    
    train_loss_name = ' '
    if cfg.train.loss == 'CrossEntropyLoss':
        train_loss_name = 'CrossEntropyLoss'
        train_loss = nn.CrossEntropyLoss()
    elif cfg.train.loss == 'LabelSmoothingLoss':
        train_loss_name = 'LabelSmoothingLoss'
        train_loss = LabelSmoothingLoss(n_classes=cfg.train.classes, 
                                        epsilon=cfg.aug.smooth_eps)
    elif cfg.train.loss == 'FocalLoss':
        train_loss_name = 'FocalLoss'
        train_loss = FocalLoss(n_classes=cfg.train.classes,
                               gamma=cfg.aug.gamma,
                               alpha=cfg.aug.alpha)
    else:
        raise ValueError(
            'the loss name you setted is not supported yet')
    
    if dist.get_rank() == 0:
        ic(train_loss_name)
    
    return train_loss

def build_aug_loss(cfg):

    aug_loss_name = ' '
    extra_aug_name = ' '
    if cfg.aug.mixup:
        extra_aug_name = 'Mixup'
        aug_loss_name = 'MixLoss'
        aug_loss = MixLoss()
    elif cfg.aug.cutmix:
        extra_aug_name = 'CutMix'
        aug_loss_name = 'MixLoss'
        aug_loss = MixLoss()
    elif cfg.aug.ricap:
        extra_aug_name = 'Ricap'
        aug_loss_name = 'RicapLoss'
        aug_loss = RicapLoss()
    elif cfg.aug.d_cutout:
        raise NotImplementedError
    else:
        aug_loss_name = 'CrossEntropyLoss'
        aug_loss = nn.CrossEntropyLoss()
    if dist.get_rank() == 0:
        ic(aug_loss_name)
        ic(extra_aug_name)
    return aug_loss