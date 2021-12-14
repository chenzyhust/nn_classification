#!/usr/bin/env	python3
""" build optim code
author: chenzy
date:   2021.2.18
"""
import torch
import torch.distributed as dist
from icecream import ic
from .sam import SAMSGD
from .lars import LARSOptimizer
from .sgd_gc import SGD_GCC, SGD_GC, SGDW, SGDW_GCC, SGDW_GC 

__all__ = ['build_optim']

def build_optim(cfg, net):

    optim_name = ' '
    if cfg.train.optim == 'sgd':
        optim_name = 'sgd'
        optimizer = torch.optim.SGD(net.parameters(), 
                                    lr=cfg.train.lr, 
                                    momentum=0.9, 
                                    weight_decay=5e-4)
    elif cfg.train.optim == 'adam':
        optim_name = 'adam'
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=cfg.train.lr)
    elif cfg.train.optim == 'sam':
        optim_name = 'sam'
        optimizer = SAMSGD(net.parameters(), 
                           lr=cfg.train.lr,
                           rho=0.05)
    elif cfg.train.optim == 'sgd_gc':
        optim_name = 'sgd_gc'
        optimizer = SGD_GC(net.parameters(), 
                           lr=cfg.train.lr, 
                           momentum=0.9, 
                           weight_decay=5e-4)
    elif cfg.train.optim == 'sgd_gcc':
        optim_name = 'sgd_gcc'
        optimizer = SGD_GCC(net.parameters(), 
                            lr=cfg.train.lr,
                            momentum=0.9, 
                            weight_decay=5e-4)
    elif cfg.train.optim == 'lars':
        optim_name = 'lars'
        optimizer = LARSOptimizer(params,
                                  lr=cfg.train.lr,
                                  momentum=0.9,
                                  eps=1e-9,
                                  thresh=-e-2)
    else:
        raise ValueError(
            'the optimizer name you setted is not supported yet')
    if dist.get_rank() == 0:
        ic(optim_name)
    
    return optimizer