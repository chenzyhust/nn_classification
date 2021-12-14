#!/usr/bin/env	python3

""" dataset code
author: chenzy
date:   2021.2.18
"""
import os
import torch
import cv2
import numpy as np
import torch.distributed as dist

from torch.utils import data
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from icecream import ic
from .transforms import augmix

def bgr_loader(path):
	return cv2.imread(path, cv2.IMREAD_COLOR)

class ImageList(Dataset):
    def __init__(self, 
                img_list, 
                img_transform=None):
        imgs = [[val.split(" ")[0], np.array(int(val.split(" ")[1]))]  for val in img_list]
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders"))

        self.imgs = imgs
        self.img_transform = img_transform
        self.loader = bgr_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

class AugMixDataset(Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, cfg, dataset, preprocess):
        self.cfg = cfg
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = cfg.aug.no_jsd

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return augmix(self.cfg, x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), augmix(self.cfg, x, self.preprocess),
                        augmix(self.cfg, x, self.preprocess))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)

def load_dataset_train(cfg, 
                       img_list, 
                       train_transforms,
                       num_workers, 
                       batch_size, 
                       shuffle=False, 
                       drop_last=False):
    if cfg.aug.augmix:
        transform1, transform2 = train_transforms
        train_transform = transforms.Compose(transform1)
        preprocess = transforms.Compose(transform2)
        train_dataset = ImageList(img_list=img_list, 
                                  img_transform=train_transform)
        train_dataset = AugMixDataset(cfg, train_dataset, preprocess)
    else:
        train_transforms = transforms.Compose(train_transforms)
        train_dataset = ImageList(img_list=img_list, 
                                  img_transform=train_transforms)
    if dist.get_rank() == 0:
        ic(train_transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = data.DataLoader(
        dataset=train_dataset, num_workers=num_workers, batch_size=batch_size,
        shuffle=shuffle, sampler=train_sampler, drop_last=drop_last)
    
    return train_loader

def load_dataset_test(cfg,
                      img_list,
                      num_workers,
                      batch_size,
                      shuffle=False,
                      drop_last=False):
    test_transforms = transforms.Compose(cfg.dataset.test_transforms)
    if dist.get_rank() == 0:
        ic(test_transforms)
    test_dataset = ImageList(img_list=img_list,
                             img_transform=test_transforms)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    test_loader = data.DataLoader(
        dataset=test_dataset, num_workers=num_workers, batch_size=batch_size,
        shuffle=shuffle, sampler=test_sampler, drop_last=drop_last)
    
    return test_loader


