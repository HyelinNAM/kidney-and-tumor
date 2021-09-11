import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset,ConcatDataset
from pycocotools.coco import COCO

from utils import define_transform

class KidneyDataset(Dataset):

    def __init__(self, data_dir = './fold/train.json', mode='train', transform=None, crop = False, new_norm = False):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        self.crop = crop
        self.new_norm = new_norm

        self.coco = COCO(data_dir)
        
    def __getitem__(self, index):

        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # load image

        if self.new_norm:
            img = cv2.imread(os.path.join('./fold/train_pre', image_infos['file_name']))

        else:
            img = cv2.imread(os.path.join('./fold/train', image_infos['file_name']))

        img = img.astype(np.float32)

        if self.mode == 'train' or 'val':
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # background -> 0
            masks = np.zeros((image_infos['height'], image_infos['width']))

            for i in range(len(anns)): # annotations 개수만큼
                pixel_value = anns[i]['category_id']
                masks = np.maximum(self.coco.annToMask(anns[i]) * pixel_value, masks)

            masks = np.array(masks, dtype=np.float32)
        
            # for albumentation
            if self.transform is not None:
                transformed = self.transform(image=img, mask=masks)
                img = transformed['image']
                masks = transformed['mask']

            return img, masks, image_infos

        elif self.mode == 'test':
            pass

    def __len__(self):
        return len(self.coco.getImgIds())

def make_foldset(fold, crop = False, new_norm= False):
    # fold - 0,1,2,3

    j0 = './fold/annotation_0.json'
    j1 = './fold/annotation_1.json'
    j2 = './fold/annotation_2.json'
    j3 = './fold/annotation_3.json'

    print('making dataset...')

    train_transform = define_transform(mode='train', crop=crop)
    val_transform = define_transform(mode='test', crop=crop)
    
    if fold == 0:
        t1 = KidneyDataset(data_dir = j1, mode = 'train', transform=train_transform, new_norm = new_norm)
        t2 = KidneyDataset(data_dir = j2, mode= 'train', transform=train_transform,  new_norm = new_norm)
        t3 = KidneyDataset(data_dir = j3, mode= 'train', transform=train_transform,  new_norm = new_norm)

        valset = KidneyDataset(data_dir=j0, mode='val', transform=val_transform, new_norm = new_norm)

    elif fold == 1:
        t1 = KidneyDataset(data_dir = j0, mode = 'train', transform=train_transform, new_norm = new_norm)
        t2 = KidneyDataset(data_dir = j2, mode= 'train', transform=train_transform, new_norm = new_norm)
        t3 = KidneyDataset(data_dir = j3, mode= 'train', transform=train_transform, new_norm = new_norm)

        valset = KidneyDataset(data_dir=j1, mode='val', transform=val_transform, new_norm = new_norm)
    
    elif fold == 2:
        t1 = KidneyDataset(data_dir = j0, mode = 'train', transform=train_transform, new_norm = new_norm)
        t2 = KidneyDataset(data_dir = j1, mode= 'train', transform=train_transform, new_norm = new_norm)
        t3 = KidneyDataset(data_dir = j3, mode= 'train', transform=train_transform, new_norm = new_norm)

        valset = KidneyDataset(data_dir=j2, mode='val', transform=val_transform, new_norm = new_norm)
    
    else:
        t1 = KidneyDataset(data_dir = j0, mode = 'train', transform=train_transform, new_norm = new_norm)
        t2 = KidneyDataset(data_dir = j1, mode= 'train', transform=train_transform, new_norm = new_norm)
        t3 = KidneyDataset(data_dir = j2, mode= 'train', transform=train_transform, new_norm = new_norm)

        valset = KidneyDataset(data_dir=j3, mode='val', transform=val_transform, new_norm = new_norm)

    return ConcatDataset([t1,t2,t3]), valset