import os
from datetime import datetime
from yacs.config import CfgNode as CN
from torchvision import transforms

_C = CN()

# global settings
_C.workspace           = "/skin_data/models/0324/mobilenetv2" # directory to save weights file
_C.date_format         = '%A_%d_%B_%Y_%Hh_%Mm_%Ss' # time of we run the script
_C.time_now            = datetime.now().strftime(_C.date_format)
_C.net                 = "mobilenet_v2" # net type
_C.local_rank          = 0
_C.device              = 'cuda'      # cuda or cpu
_C.sync_bn             = False           
_C.apex                = False
# 'O0' fp32，'O1'、'O2' fp16
_C.opt_level           = 'O1'

# dataset parameters
_C.dataset                   = CN()
_C.dataset.train_path        = "/skin_data/train.txt"
_C.dataset.test_path         = "/skin_data/test.txt"
_C.dataset.img_width         = 64
_C.dataset.img_height        = 64
_C.dataset.num_workers       = 4
_C.dataset.train_transforms  = [
  transforms.ToPILImage(),
  transforms.Resize([_C.dataset.img_width, _C.dataset.img_height]),
#   transforms.RandomCrop((_C.dataset.img_width, _C.dataset.img_height), 16),
  transforms.RandomHorizontalFlip(p=0.5),
#   transforms.RandomVerticalFlip(p=0.5),
#   transforms.RandomRotation(15),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]
_C.dataset.test_transforms   = [
  transforms.ToPILImage(),
  transforms.Resize([64, 64]),
  #transforms.CenterCrop((_C.dataset.img_width, _C.dataset.img_height)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

# train parameters 
_C.train = CN()
_C.train.classes     = 2             # num of classes
_C.train.epoches     = 50              # total training epoches
_C.train.batch       = 512               # batch size dp : ALL=BATCH  dist : ALL=BATCH * world_size
_C.train.warm        = 2                 # warm up epoches
_C.train.optim       = "sgd"             # optim type 
_C.train.lr          = 0.4               # initial learning rate
_C.train.scheduler   = "cosine"          # learning rate adjust type step or cosine
_C.train.steps       = [60, 120, 160]
_C.train.resume      = False             # resume training
_C.train.resume_path = ''                # resume checkpoint path
_C.train.pretrained  = True              # load imagenet pretrained weights
_C.train.gpu_id      = "0,1,2,3"         # id(s) for CUDA_VISIBLE_DEVICES
_C.train.cudnn       = False             # cudnn.benchmark 
_C.train.detem       = True              # cudnn.deterministic
_C.train.seed        = 0
_C.train.swa         = False
_C.train.swa_start   = 161
_C.train.swa_lr      = 0.02
_C.train.swa_cepo    = 1                  # SWA model collection frequency/cycle length in epochs (default: 1)
_C.train.loss        = "CrossEntropyLoss" # ["CrossEntropyLoss", "LabelSmoothingLoss","FocalLoss", "CenterLoss"]

# aug strategy
_C.aug = CN()
_C.aug.prob           = 0.5         # probability of data aug
#                     : augmix
_C.aug.augmix         = False
_C.aug.mixture_width  = 3
_C.aug.mixture_depth  = 1
_C.aug.prob_coeff     = 1
_C.aug.severity       = 1
_C.aug.no_jsd         = True
#                     : mixup
_C.aug.mixup          = False       # whether to use mixup
_C.aug.mixup_a        = 1           # mixup alpha paramter
#                     : cutout & dual cutout
_C.aug.cutout         = False       # whether to use cutout
_C.aug.d_cutout       = False       # whether to use dualcutout
_C.aug.n_holes        = 1           # number of holes to cut out from image
_C.aug.length         = 8           # length of the holes
_C.aug.dcut_alpha     = 0.1         # dual cutout loss parameter
#                     : random erasing
_C.aug.r_erasing      = False       # whether to use cutout
_C.aug.m_attempt      = 20          # num of max attempt
_C.aug.a_rat_range    = [0.02, 0.4] # area ratio range
_C.aug.m_asp_ratio    = 0.3         # min aspect ration
#                     : cutmix
_C.aug.cutmix         = True       # whether to use cutmix
_C.aug.cutmix_beta    = 1           # hyperparameter beta of cutmix
#                     : ricap
_C.aug.ricap          = False       # whether to use ricap
_C.aug.ricap_beta     = 0.3         # hyperparameter beta of ricap
#                     : label_smoothing
_C.aug.smooth_eps     = 0.1         # label_smoothing epsilon
#                     : focal_loss
_C.aug.gamma          = 2.0         # focal loss gamma
_C.aug.alpha          = 0.25        # focal loss alpha

# test parameter
_C.test = CN()
_C.test.weight  = "./checkpoint/resnet50/Friday_01_January_2021_14h_45m_41s/resnet50-190-best.pth"
                                   # weight file path
_C.test.batch   = 128              # batch size
_C.test.gpu_id  = "0,1,2,3"        # id(s) for CUDA_VISIBLE_DEVICES

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
