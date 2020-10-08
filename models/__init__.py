import torch
from config import cfg
import torchvision


def get_vgg16_for_cifar():
    from models.cifar.vgg import VGG
    return VGG('VGG16', cfg.model.num_class)

def get_resnet50_for_imagenet():
    # from models.imagenet.resnet50 import Resnet50
    # return Resnet50(cfg.model.num_class)
    return torchvision.models.resnet50(pretrained=cfg.model.pretrained)

def get_resnet56():
    from models.cifar.resnet56 import resnet56
    return resnet56(cfg.model.num_class)

def get_wideresnet():
    import models.cifar.wideresnet as wrn
    return wrn.Model(num_classes=cfg.model.num_class)

def get_resnet56m():
    import models.cifar.resnet56m as rs56
    return rs56.resnet56_cifar(input_resolution=cfg.model.resolution)

def get_resnet50():
    import models.cifar.resnet50 as rs50
    return rs50.Model(num_classes=100, input_size=32)

def get_mobilenetv2():
    import models.cifar.mobilenetv2 as mn
    return mn.Model(num_classes=100, input_size=32)

def get_model():
    pair = {
        'cifar.vgg16': get_vgg16_for_cifar,
        'resnet50': get_resnet50_for_imagenet,
        'cifar.resnet56': get_resnet56,
        'wideresnet': get_wideresnet,
        'resnet56m': get_resnet56m,
        'resnet50c': get_resnet50,
        'mobilenetv2': get_mobilenetv2
    }

    model = pair[cfg.model.name]()

    if cfg.base.checkpoint_path != '':
        print('restore checkpoint: ' + cfg.base.checkpoint_path)
        model.load_state_dict(torch.load(cfg.base.checkpoint_path, map_location='cpu' if not cfg.base.cuda else 'cuda'))

    if cfg.base.cuda:
        model = model.cuda()

    if cfg.base.multi_gpus:
        model = torch.nn.DataParallel(model)
    return model
