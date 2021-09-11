import os
import random
from albumentations.augmentations.crops.transforms import CenterCrop
from albumentations.augmentations.transforms import Normalize
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def set_seed(random_seed=28):
    print('Setting seed...')

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def collate_fn(batch):
    return tuple(zip(*batch))

def define_transform(mode='train', crop=True):
    if mode == 'train':
        if crop:
            transform = A.Compose([
                A.CenterCrop(384,384,always_apply=True),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2()
            ])

        else:
            transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2()
            ])

    else: # val or test
        if crop:
            transform = A.Compose([
                A.CenterCrop(384,384,always_apply=True),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Normalize(),
                ToTensorV2()
            ])

    return transform

def define_scheduler(s_name, optimizer, total, lr):
    if s_name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max = total, eta_min = lr * 0.01)

    elif s_name == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

    elif s_name == 'step':
        return lr_scheduler.MultiStepLR(optimizer, milestones=[30,40] , gamma = 0.1) # 50
    
    elif s_name == 'nope':
        return

def load_ckpt(checkpoint_path, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, checkpoint['epoch'], scheduler


def save_model(checkpoint, saved_dir='./statedict', file_name = 'efficientnet_baseline.pt'):

    if not os.path.isdir(saved_dir):
        os.mkdir(saved_dir)

    output_path = os.path.join(saved_dir, file_name)

    torch.save(checkpoint, output_path)