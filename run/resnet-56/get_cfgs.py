''' setting before run. every notebook should include this code. '''
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys

_r = os.getcwd().split('/')
_p = '/'.join(_r[:_r.index('gate-decorator-pruning')+1])
print('Change dir from %s to %s' % (os.getcwd(), _p))
os.chdir(_p)
sys.path.append(_p)

model_name = "resnet56m"
res = 20
from config import parse_from_dict
parse_from_dict({
    "base": {
        "task_name": "{}_cifar100_ticktock_{}".format(model_name, res),
        "model_saving_interval": 1,
        "cuda": True,
        "seed": 1995,
        "checkpoint_path": "",
        "epoch": 0,
        "multi_gpus": True,
        "fp16": False
    },
    "model": {
        "name": "{}".format(model_name),
        "num_class": 100,
        "pretrained": False,
        "resolution": res
    },
    "train": {
        "trainer": "normal",
        "max_epoch": 200,
        "optim": "sgd",
        # "steplr": [
        #     [80, 0.1],
        #     [120, 0.01],
        #     [160, 0.001]
        # ],
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "nesterov": True
    },
    "data": {
        "type": "cifar100",
        "shuffle": True,
        "batch_size": 128,
        "test_batch_size": 128,
        "num_workers": 4
    },
    "loss": {
        "criterion": "softmax"
    },
    "gbn": {
        "sparse_lambda": 1e-3,
        "flops_eta": 0,
        "lr_min": 1e-3,
        "lr_max": 1e-2,
        "tock_epoch": 10, #10
        "T": 10,#10
        "p": 0.002
    }
})
from config import cfg

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from logger import logger
from main import set_seeds, recover_pack, _step_lr, _sgdr
from models import get_model
from utils import dotdict

from prune.universal import Meltable, GatedBatchNorm2d, Conv2dObserver, IterRecoverFramework, FinalLinearObserver
from prune.utils import analyse_model, finetune

def get_cfg(p):
    print(p)

    set_seeds()
    pack = recover_pack()

    GBNs = GatedBatchNorm2d.transform(pack.net)
    for gbn in GBNs:
        gbn.extract_from_bn()

    model_dict = torch.load('logs/{}_cifar100_ticktock_{}/{}.ckp'.format(model_name, res, p), map_location='cpu' if not cfg.base.cuda else 'cuda')
    # model_dict = torch.load('logs/wideresnet_cifar100_baseline/ckp.189.torch'.format(p), map_location='cpu' if not cfg.base.cuda else 'cuda')
    pack.net.module.load_state_dict(model_dict)


    _ = Conv2dObserver.transform(pack.net.module)
    pack.net.module.fc = FinalLinearObserver(pack.net.module.fc)
    # pack.net.module.classifier[0] = FinalLinearObserver(pack.net.module.classifier[0])
    Meltable.observe(pack, 0.001)
    Meltable.melt_all(pack.net)

    cfgs = []
    for m in pack.net.named_modules():
        if isinstance(m[1], nn.BatchNorm2d):
            cfgs.append(m[1].num_features)

    BASE_FLOPS, BASE_PARAM = analyse_model(pack.net.module, torch.randn(1, 3, res, res).cuda())

    save_dir = 'logs/{}_cifar100_ticktock_{}/cfgs'.format(model_name, res)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(save_dir + '/' + str(p)+'.cfg', 'w')
    f.write(str(cfgs))
    f.write('\n')
    f.write('%.3f MFLOPS\n' % (BASE_FLOPS / 1e6))
    f.write('%.3f M Params' % (BASE_PARAM / 1e6))
    f.write('\n\n')
    f.write(str(pack.net))
    f.close()


for p in range(40, 100, 5):
    get_cfg(p)