"""
 * Copyright (C) 2019 Zhonghui You
 * If you are using this code in your research, please cite the paper:
 * Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.
"""

from time import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
from config import cfg
import torch.optim as optim


FINISH_SIGNAL = 'finish'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class NormalTrainer():
    def __init__(self):
        self.use_cuda = cfg.base.cuda

    def test(self, pack, topk=(1,)):
        pack.net.eval()
        loss_acc, correct, total = 0.0, 0.0, 0.0
        hub = [[] for i in range(len(topk))]

        for data, target in pack.test_loader:
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                output = pack.net(F.interpolate(data, (cfg.model.resolution, cfg.model.resolution), mode='bilinear', align_corners=True))
                loss_acc += pack.criterion(output, target).data.item()
                acc = accuracy(output, target, topk)
                for acc_idx, score in enumerate(acc):
                    hub[acc_idx].append(score[0].item())

        loss_acc /= len(pack.test_loader)
        info = {
            'test_loss': loss_acc
        }
        
        for acc_idx, k in enumerate(topk):
            info['acc@%d' % k] = np.mean(hub[acc_idx])

        return info

    def train(self, pack, loss_hook=None, iter_hook=None, update=True, mute=False, acc_step=1):
        pack.net.train()
        loss_acc, correct_acc, total = 0.0, 0.0, 0.0
        begin = time()

        pack.optimizer.zero_grad()
        with tqdm(total=len(pack.train_loader), disable=mute) as pbar:
            total_iter = len(pack.train_loader)
            for cur_iter, (data, label) in enumerate(pack.train_loader):
                if iter_hook is not None:
                    signal = iter_hook(cur_iter, total_iter)
                    if signal == FINISH_SIGNAL:
                        break
                if self.use_cuda:
                    data, label = data.cuda(), label.cuda()
                data = Variable(data, requires_grad=False)
                label = Variable(label)

                logits = pack.net(data)
                loss = pack.criterion(logits, label)
                if loss_hook is not None:
                    additional = loss_hook(data, label, logits)
                    loss += additional
                loss = loss / acc_step
                loss.backward()

                if (cur_iter + 1) % acc_step == 0:
                    if update:
                        pack.optimizer.step()
                        lr = self.adjust_learning_rate(pack)
                    pack.optimizer.zero_grad()

                loss_acc += loss.item()
                pbar.update(1)

        info = {
            'train_loss': loss_acc / len(pack.train_loader),
            'epoch_time': time() - begin
        }
        return info

    def adjust_learning_rate(self, pack):
        if pack.optimizer is None:
            if cfg.train.optim == 'sgd' or cfg.train.optim is None:
                pack.optimizer = optim.SGD(
                    pack.net.parameters(),
                    # lr=1,
                    lr=0.1,
                    momentum=cfg.train.momentum,
                    weight_decay=cfg.train.weight_decay,
                    nesterov=cfg.train.nesterov
                )
            else:
                print('WRONG OPTIM SETTING!')
                assert False
            # pack.lr_scheduler = optim.lr_scheduler.LambdaLR(pack.optimizer, get_lr_func())
            pack.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pack.optimizer, len(pack.train_loader)*cfg.train.max_epoch)

        # pack.lr_scheduler.step(epoch)
        pack.lr_scheduler.step()
        return pack.optimizer.param_groups[0]['lr'] #pack.lr_scheduler.get_lr()

class MutualTrainer():
    def __init__(self):
        self.use_cuda = cfg.base.cuda

    def test(self, pack, topk=(1,)):
        pack.net.eval()
        loss_acc, correct, total = 0.0, 0.0, 0.0
        hub = [[] for i in range(len(topk))]

        for data, target in pack.test_loader:
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                output = pack.net(F.interpolate(data, (cfg.model.resolution[0], cfg.model.resolution[0]), mode='bilinear', align_corners=True))
                loss_acc += pack.criterion(output, target).data.item()
                acc = accuracy(output, target, topk)
                for acc_idx, score in enumerate(acc):
                    hub[acc_idx].append(score[0].item())

        loss_acc /= len(pack.test_loader)
        info = {
            'test_loss': loss_acc
        }
        
        for acc_idx, k in enumerate(topk):
            info['acc@%d' % k] = np.mean(hub[acc_idx])

        return info

    def train(self, pack, loss_hook=None, iter_hook=None, update=True, mute=False, acc_step=1):
        pack.net.train()
        loss_acc, correct_acc, total = 0.0, 0.0, 0.0
        begin = time()

        pack.optimizer.zero_grad()
        with tqdm(total=len(pack.train_loader), disable=mute) as pbar:
            total_iter = len(pack.train_loader)
            for cur_iter, (data, label) in enumerate(pack.train_loader):
                if iter_hook is not None:
                    signal = iter_hook(cur_iter, total_iter)
                    if signal == FINISH_SIGNAL:
                        break
                if self.use_cuda:
                    # data, label = data.cuda(), label.cuda()
                    data = [_data.cuda() for _data in data]
                    label = label.cuda(non_blocking=True)
                # data = Variable(data, requires_grad=False)
                # label = Variable(label)

                logits = pack.net(data[0])
                max_output_detach = logits.detach()
                loss = pack.criterion(logits, label)
                if loss_hook is not None:
                    additional = loss_hook(data, label, logits)
                    loss += additional
                loss = loss / acc_step
                loss.backward()
                # *********************
                for idx in range(1, len(cfg.model.resolution)):
                    output = pack.net(data[idx])
                    loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1), F.softmax(max_output_detach, dim=1))
                    loss.backward()
                # *********************

                if (cur_iter + 1) % acc_step == 0:
                    if update:
                        pack.optimizer.step()
                        lr = self.adjust_learning_rate(pack)
                    pack.optimizer.zero_grad()

                loss_acc += loss.item()
                pbar.update(1)

        info = {
            'train_loss': loss_acc / len(pack.train_loader),
            'epoch_time': time() - begin
        }
        return info

    def adjust_learning_rate(self, pack):
        if pack.optimizer is None:
            if cfg.train.optim == 'sgd' or cfg.train.optim is None:
                pack.optimizer = optim.SGD(
                    pack.net.parameters(),
                    # lr=1,
                    lr=0.1,
                    momentum=cfg.train.momentum,
                    weight_decay=cfg.train.weight_decay,
                    nesterov=cfg.train.nesterov
                )
            else:
                print('WRONG OPTIM SETTING!')
                assert False
            # pack.lr_scheduler = optim.lr_scheduler.LambdaLR(pack.optimizer, get_lr_func())
            pack.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pack.optimizer, len(pack.train_loader)*cfg.train.max_epoch)

        # pack.lr_scheduler.step(epoch)
        pack.lr_scheduler.step()
        return pack.optimizer.param_groups[0]['lr'] #pack.lr_scheduler.get_lr()

class MutualTrainerBN():
    def __init__(self):
        self.use_cuda = cfg.base.cuda

    def test(self, pack, topk=(1,)):
        pack.net.eval()
        loss_acc, correct, total = 0.0, 0.0, 0.0
        hub = [[] for i in range(len(topk))]

        for data, target in pack.test_loader:
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                output = pack.net(F.interpolate(data, (cfg.model.resolution[0], cfg.model.resolution[0]), mode='bilinear', align_corners=True))
                loss_acc += pack.criterion(output, target).data.item()
                acc = accuracy(output, target, topk)
                for acc_idx, score in enumerate(acc):
                    hub[acc_idx].append(score[0].item())

        loss_acc /= len(pack.test_loader)
        info = {
            'test_loss': loss_acc
        }
        
        for acc_idx, k in enumerate(topk):
            info['acc@%d' % k] = np.mean(hub[acc_idx])

        return info

    def train(self, pack, loss_hook=None, iter_hook=None, update=True, mute=False, acc_step=1):
        pack.net.train()
        loss_acc, correct_acc, total = 0.0, 0.0, 0.0
        begin = time()

        pack.optimizer.zero_grad()
        with tqdm(total=len(pack.train_loader), disable=mute) as pbar:
            total_iter = len(pack.train_loader)
            for cur_iter, (data, label) in enumerate(pack.train_loader):
                if iter_hook is not None:
                    signal = iter_hook(cur_iter, total_iter)
                    if signal == FINISH_SIGNAL:
                        break
                if self.use_cuda:
                    # data, label = data.cuda(), label.cuda()
                    data = [_data.cuda() for _data in data]
                    label = label.cuda(non_blocking=True)
                # data = Variable(data, requires_grad=False)
                # label = Variable(label)

                logits = pack.net(data[0])
                max_output_detach = logits.detach()
                loss = pack.criterion(logits, label)
                if loss_hook is not None:
                    additional = loss_hook(data, label, logits)
                    loss += additional
                loss = loss / acc_step
                loss.backward()
                # *********************
                for idx in range(1, len(cfg.model.resolution)):
                    output = pack.net(data[idx])
                    loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1), F.softmax(max_output_detach, dim=1))
                    loss.backward()
                # *********************

                if (cur_iter + 1) % acc_step == 0:
                    if update:
                        pack.optimizer.step()
                        lr = self.adjust_learning_rate(pack)
                    pack.optimizer.zero_grad()

                loss_acc += loss.item()
                pbar.update(1)

        info = {
            'train_loss': loss_acc / len(pack.train_loader),
            'epoch_time': time() - begin
        }
        return info

    def adjust_learning_rate(self, pack):
        if pack.optimizer is None:
            if cfg.train.optim == 'sgd' or cfg.train.optim is None:
                pack.optimizer = optim.SGD(
                    pack.net.parameters(),
                    # lr=1,
                    lr=0.1,
                    momentum=cfg.train.momentum,
                    weight_decay=cfg.train.weight_decay,
                    nesterov=cfg.train.nesterov
                )
            else:
                print('WRONG OPTIM SETTING!')
                assert False
            # pack.lr_scheduler = optim.lr_scheduler.LambdaLR(pack.optimizer, get_lr_func())
            pack.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pack.optimizer, len(pack.train_loader)*cfg.train.max_epoch)

        # pack.lr_scheduler.step(epoch)
        pack.lr_scheduler.step()
        return pack.optimizer.param_groups[0]['lr'] #pack.lr_scheduler.get_lr()
