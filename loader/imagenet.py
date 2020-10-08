"""
 * Copyright (C) 2019 Zhonghui You
 * If you are using this code in your research, please cite the paper:
 * Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.
"""

import torch
import torchvision
from torchvision import transforms
import os

from config import cfg

# def _get_loaders(root):
#     train_dir = os.path.join(root, 'train')
#     val_dir = os.path.join(root, 'val')

#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])

#     train_dataset = torchvision.datasets.ImageFolder(
#         train_dir,
#         transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]))
    
#     val_dataset = torchvision.datasets.ImageFolder(
#         val_dir,
#         transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             normalize,
#         ]))
    
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=cfg.data.batch_size,
#         shuffle=cfg.data.shuffle, 
#         num_workers=cfg.data.num_workers,
#         pin_memory=True
#     )

#     val_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=cfg.data.test_batch_size,
#         shuffle=False,
#         num_workers=cfg.data.num_workers,
#         pin_memory=True
#     )

#     return train_loader, val_loader

from torchvision import datasets, transforms
from loader.transforms import Lighting
def _get_imagenet():
    image_size = 224
    image_resize = 256
    dataset_dir = "/data/ImageNet/ILSVRC/Data/CLS-LOC"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    crop_scale = 0.08
    jitter_param = 0.4
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(crop_scale, 1.0)),
        transforms.ColorJitter(
            brightness=jitter_param, contrast=jitter_param,
            saturation=jitter_param),
        Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        # InputList(FLAGS.resolution_list),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(image_resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    train_set = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=train_transforms)
    val_set = datasets.ImageFolder(os.path.join(dataset_dir, 'val'), transform=val_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.data.batch_size, shuffle=True,
        pin_memory=True, num_workers=cfg.data.num_workers,
        drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.data.batch_size, shuffle=False,
        pin_memory=True, num_workers=cfg.data.num_workers,
        drop_last=False)

    return train_loader, val_loader

def get_imagenet():
    # return _get_loaders('./data/imagenet12')
    return _get_imagenet()
