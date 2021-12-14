#!/usr/bin/env bash
python -m torch.distributed.launch --nproc_per_node=2  --master_port=25900 train.py