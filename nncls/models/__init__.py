#!/usr/bin/env	python3
""" build model code
author: chenzy
date:   2021.2.18
"""
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import imagenet
import torch.distributed as dist
from icecream import ic
from torch.hub import load_state_dict_from_url
from torch.nn.parallel import DistributedDataParallel as NativeDDP

__all__ = ['build_model']

SERVER = "http://10.128.2.79:7000/weights"

def build_model(cfg):

    net = getattr(imagenet, cfg.net)(cfg.train.classes)
    if cfg.train.pretrained:
        weights_dir = os.path.join(cfg.workspace, "modelweight")
        weight_path = weights_dir + "/{}.pth".format(cfg.net)
        if os.path.exists(weight_path):
            pretrained_dict = torch.load(weight_path)
        else:
            if dist.get_rank() == 0:
                if not os.path.exists(weights_dir):
                    os.makedirs(weights_dir)
            url = SERVER + "/{}.pth".format(cfg.net)
            if dist.get_rank() == 0:
                print(url)
            pretrained_dict = load_state_dict_from_url(url, model_dir=weights_dir)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                        k in model_dict and 'classifier' not in k and 'fc' not in k}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    
    net.to(cfg.device)
    model_name = cfg.net
    if dist.get_rank() == 0:
        ic(model_name)

    if not cfg.apex and dist.is_available():
        local_rank = cfg.local_rank
        if cfg.sync_bn:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = net.to(local_rank)
        net = NativeDDP(net,
                        device_ids=[local_rank],
                        output_device=local_rank)
    
    return net