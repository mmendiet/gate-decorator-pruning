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

from config import parse_from_dict
parse_from_dict({
    "base": {
        "task_name": "resnet50pre_cifar100_ticktock",
        "model_saving_interval": 1,
        "cuda": True,
        "seed": 1995,
        "checkpoint_path": "",
        "epoch": 0,
        "multi_gpus": True,
        "fp16": False
    },
    "model": {
        "name": "resnet50",
        "num_class": 1000,
        "pretrained": False,
        "resolution":224
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
        "type": "imagenet",
        "shuffle": True,
        "batch_size": 256,
        "test_batch_size": 256,
        "num_workers": 16
    },
    "loss": {
        "criterion": "softmax"
    },
    "gbn": {
        "sparse_lambda": 1e-3,
        "flops_eta": 0,
        "lr_min": 1e-3,#3
        "lr_max": 1e-2,#2
        "tock_epoch": 10,
        "T": 10,
        "p": 0.0002
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

set_seeds()
pack = recover_pack()

# model_dict = torch.load('logs/resnet56m_cifar100_baseline_{}4/ckp.398.torch'.format(res), map_location='cpu' if not cfg.base.cuda else 'cuda')
# pack.net.module.load_state_dict(model_dict)

GBNs = GatedBatchNorm2d.transform(pack.net)
for gbn in GBNs:
    gbn.extract_from_bn()

pack.optimizer = optim.SGD(
    pack.net.parameters() ,
    lr=2e-3,
    momentum=cfg.train.momentum,
    weight_decay=cfg.train.weight_decay,
    nesterov=cfg.train.nesterov
)
print(cfg.base.task_name)
import uuid

def bottleneck_set_group(net):
    layers = [
        net.module.layer1,
        net.module.layer2,
        net.module.layer3,
        net.module.layer4
    ]
    for m in layers:
        masks = []
        if m == net.module.layer1:
            masks.append(pack.net.module.bn1)
        for mm in m.modules():
            if mm.__class__.__name__ == 'BasicBlock':
                if mm.downsample:
                    masks.append(mm.downsample._modules['1'])
                masks.append(mm.bn2)

        group_id = uuid.uuid1()
        for mk in masks:
            mk.set_groupid(group_id)

# def bottleneck_set_group(net):
#     layers = [3,7,13,16]
#     masks = []
#     for idx, m in enumerate(net.module.features):
#         # if idx == 0:
#             # masks.append(m[1])
#         if m.__class__.__name__ == 'Block':
#             if not m.residual_connection:
#                 masks.append(m.shortcut._modules['1'])
#             masks.append(m.body[7])
#         if idx in layers:
#             group_id = uuid.uuid1()
#             for mk in masks:
#                 mk.set_groupid(group_id)
#             masks = []

# def bottleneck_set_group(net):
#     layers = [2,5,9,12,15]
#     masks = []
#     for idx, m in enumerate(net.module.features):
#         # if idx == 0:
#             # masks.append(m[1])
#         if m.__class__.__name__ == 'InvertedResidual':
#             if m.residual_connection:
#                 masks.append(m.body[7])
#         if idx in layers:
#             group_id = uuid.uuid1()
#             for mk in masks:
#                 mk.set_groupid(group_id)
#             masks = []

#     #depthwise
#     masks = []
#     for idx, m in enumerate(net.module.features):
#         if idx == 0:
#             masks.append(m[1])
#         if m.__class__.__name__ == 'InvertedResidual':
#             for i in range(0, len(m.body)-1):
#                 if isinstance(m.body[i], nn.Conv2d):
#                     if m.body[i].groups > 1:
#                         masks.append(m.body[i+1])
#                         break
#                     else:
#                         masks.append(m.body[i+1])
#             group_id = uuid.uuid1()
#             if len(masks) > 1:
#                 for mk in masks:
#                     mk.set_groupid(group_id)
#             masks = []

bottleneck_set_group(pack.net)

def clone_model(net):
    model = get_model()
    gbns = GatedBatchNorm2d.transform(model.module)
    model.load_state_dict(net.state_dict())
    return model, gbns

cloned, _ = clone_model(pack.net)
BASE_FLOPS, BASE_PARAM = analyse_model(cloned.module, torch.randn(1, 3, cfg.model.resolution, cfg.model.resolution).cuda())
print('%.3f MFLOPS' % (BASE_FLOPS / 1e6))
print('%.3f M' % (BASE_PARAM / 1e6))
del cloned

def eval_prune(pack):
    cloned, _ = clone_model(pack.net)
    _ = Conv2dObserver.transform(cloned.module)
    # cloned.module.classifier[0] = FinalLinearObserver(cloned.module.classifier[0])
    cloned.module.fc = FinalLinearObserver(cloned.module.fc)
    cloned_pack = dotdict(pack.copy())
    cloned_pack.net = cloned
    Meltable.observe(cloned_pack, 0.001)
    Meltable.melt_all(cloned_pack.net)
    flops, params = analyse_model(cloned_pack.net.module, torch.randn(1, 3, cfg.model.resolution, cfg.model.resolution).cuda())
    del cloned
    del cloned_pack
    
    return flops, params

pack.trainer.test(pack)

pack.tick_trainset = pack.train_loader
prune_agent = IterRecoverFramework(pack, GBNs, sparse_lambda = cfg.gbn.sparse_lambda, flops_eta = cfg.gbn.flops_eta, minium_filter = 3)

LOGS = []
flops_save_points = set([95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5])

iter_idx = 0
prune_agent.tock(lr_min=cfg.gbn.lr_min, lr_max=cfg.gbn.lr_max, tock_epoch=cfg.gbn.tock_epoch)
while True:
    left_filter = prune_agent.total_filters - prune_agent.pruned_filters
    num_to_prune = int(left_filter * cfg.gbn.p)
    info = prune_agent.prune(num_to_prune, tick=True, lr=cfg.gbn.lr_min)
    flops, params = eval_prune(pack)
    info.update({
        'flops': '[%.2f%%] %.3f MFLOPS' % (flops/BASE_FLOPS * 100, flops / 1e6),
        'param': '[%.2f%%] %.3f M' % (params/BASE_PARAM * 100, params / 1e6)
    })
    LOGS.append(info)
    print('Iter: %d,\t FLOPS: %s,\t Param: %s,\t Left: %d,\t Pruned Ratio: %.2f %%,\t Train Loss: %.4f,\t Test Acc: %.2f' % 
          (iter_idx, info['flops'], info['param'], info['left'], info['total_pruned_ratio'] * 100, info['train_loss'], info['after_prune_test_acc']))
    
    iter_idx += 1
    if iter_idx % cfg.gbn.T == 0:
        print('Tocking:')
        prune_agent.tock(lr_min=cfg.gbn.lr_min, lr_max=cfg.gbn.lr_max, tock_epoch=cfg.gbn.tock_epoch)

    flops_ratio = flops/BASE_FLOPS * 100
    for point in [i for i in list(flops_save_points)]:
        if flops_ratio <= point:
            torch.save(pack.net.module.state_dict(), './logs/{}/{}.ckp'.format(cfg.base.task_name, point))
            flops_save_points.remove(point)

    if len(flops_save_points) == 0:
        break