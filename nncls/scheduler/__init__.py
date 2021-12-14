#!/usr/bin/env	python3
""" build scheduler code
author: chenzy
date:   2021.2.18
"""
import torch.optim.lr_scheduler as optim
import torch.distributed as dist
from icecream import ic
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['build_scheduler', 'WarmUpLR']

def build_scheduler(cfg, optimizer):
    
    scheduler_name = ' '
    if cfg.train.scheduler == 'step':
        scheduler_name = 'step'
        scheduler = optim.MultiStepLR(optimizer, 
                                      milestones=cfg.train.steps, 
                                      gamma=0.2)
    elif cfg.train.scheduler == 'cosine':
        scheduler_name = 'cosine'
        scheduler = optim.CosineAnnealingLR(optimizer, 
                                            cfg.train.epoches)
    else:
        raise ValueError(
            'the lr scheduler name you setted is not supported yet')
    if dist.get_rank() == 0:
       ic(scheduler_name)

    return scheduler

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]