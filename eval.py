#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""
import os

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config.config import get_cfg_defaults
from nncls.dataset import build_test_loader
from nncls.models import build_model

if __name__ == '__main__':

    cfg = get_cfg_defaults()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.test.gpu_id
    model = build_model(cfg)

    test_loader = load_test_loader(cfg)
    model.load_state_dict(torch.load(cfg.test.weight))
    print(model)
    model.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            image = image.cuda()
            label = label.cuda()

            output = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()


    print()
    print("Top 1 acc: ", correct_1 / len(test_loader.dataset))
    print("Top 5 acc: ", correct_5 / len(test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))