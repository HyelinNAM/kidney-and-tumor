import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def define_loss(loss_name):

    if loss_name == 'bce':
        return nn.CrossEntropyLoss()

    elif loss_name == 'focal':
        return smp.losses.FocalLoss(mode='multiclass')

    elif loss_name == 'lovasz':
        return smp.losses.LovaszLoss(mode='multiclass')

    elif loss_name == 'dice':
        return smp.losses.DiceLoss(mode='multiclass')

    elif loss_name == 'combo1':
        return nn.CrossEntropyLoss(), smp.losses.DiceLoss(mode='multiclass')
    
    elif loss_name in ['combo2', 'combo3', 'combo4']:
        return smp.losses.FocalLoss(mode='multiclass'), smp.losses.DiceLoss(mode='multiclass')
