"""
 * Copyright (C) 2019 Zhonghui You
 * If you are using this code in your research, please cite the paper:
 * Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.
"""
# from config import parse_from_dict
# parse_from_dict({
#     "base": {
#         "task_name": "resnet56m_cifar100_baseline_204",
#         "model_saving_interval": 1,
#         "cuda": True,
#         "seed": 1995,
#         "checkpoint_path": "",
#         "epoch": 0,
#         "multi_gpus": True,
#         "fp16": False
#     },
#     "model": {
#         "name": "resnet56m",
#         "num_class": 100,
#         "pretrained": False,
#         "resolution": 20
#     },
#     "train": {
#         "trainer": "normal",
#         "max_epoch": 400,
#         "optim": "sgd",
#         # "steplr": [
#         #     [80, 0.1],
#         #     [120, 0.01],
#         #     [160, 0.001]
#         # ],
#         "weight_decay": 5e-4,
#         "momentum": 0.9,
#         "nesterov": True
#     },
#     "data": {
#         "type": "cifar100",
#         "shuffle": True,
#         "batch_size": 128,
#         "test_batch_size": 128,
#         "num_workers": 4
#     },
#     "loss": {
#         "criterion": "softmax"
#     },
#     "gbn": {
#         "sparse_lambda": 1e-3,
#         "flops_eta": 0,
#         "lr_min": 1e-3,
#         "lr_max": 1e-2,
#         "tock_epoch": 10,
#         "T": 10,
#         "p": 0.002
#     }
# })

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
import math

from loader import get_loader
from models import get_model
from trainer import get_trainer
from loss import get_criterion

from utils import dotdict
from config import cfg
from logger import logger


def _sgdr(epoch):
    lr_min, lr_max = cfg.train.sgdr.lr_min, cfg.train.sgdr.lr_max
    restart_period = cfg.train.sgdr.restart_period
    _epoch = epoch - cfg.train.sgdr.warm_up

    while _epoch/restart_period > 1.:
        _epoch = _epoch - restart_period
        restart_period = restart_period * 2.

    radians = math.pi*(_epoch/restart_period)
    return lr_min + (lr_max - lr_min) *  0.5*(1.0 + math.cos(radians))

def _step_lr(epoch):
    v = 0.0
    for max_e, lr_v in cfg.train.steplr:
        v = lr_v
        if epoch <= max_e:
            break
    return v

def get_lr_func():
    if cfg.train.steplr is not None:
        return _step_lr
    elif cfg.train.sgdr is not None:
        return _sgdr
    else:
        assert False

def recover_pack():
    train_loader, test_loader = get_loader()

    pack = dotdict({
        'net': get_model(),
        'train_loader': train_loader,
        'test_loader': test_loader,
        'trainer': get_trainer(),
        'criterion': get_criterion(),
        'optimizer': None,
        'lr_scheduler': None
    })

    pack.trainer.adjust_learning_rate(pack)
    return pack

def set_seeds():
    torch.manual_seed(cfg.base.seed)
    if cfg.base.cuda:
        torch.cuda.manual_seed_all(cfg.base.seed)
        torch.backends.cudnn.deterministic = True
        if cfg.base.fp16:
            torch.backends.cudnn.enabled = True
            # torch.backends.cudnn.benchmark = True
    np.random.seed(cfg.base.seed)
    random.seed(cfg.base.seed)


def main():
    set_seeds()
    pack = recover_pack()
    best_acc = 0.0
    # pack.trainer.adjust_learning_rate(pack)
    for epoch in range(cfg.base.epoch + 1, cfg.train.max_epoch + 1):
        info = pack.trainer.train(pack)
        info.update(pack.trainer.test(pack))
        lr = pack.optimizer.param_groups[0]['lr']
        info.update({'LR': lr})
        print(epoch, info)
        logger.save_record(epoch, info)
        # if epoch % cfg.base.model_saving_interval == 0:
        if info['acc@1'] > best_acc:
            logger.save_network(epoch, pack.net)
            best_acc = info['acc@1']


if __name__ == '__main__':
    main()