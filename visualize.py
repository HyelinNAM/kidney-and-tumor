import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dataset import make_foldset
from utils import collate_fn
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu" 
   
device = "cuda" if torch.cuda.is_available() else "cpu" 
   
train_dataset, val_dataset = make_foldset(fold=0)
val_loader = DataLoader(dataset= val_dataset,batch_size=1, collate_fn=collate_fn, shuffle=False)

model = smp.Unet(encoder_name='tu-efficientnetv2_rw_m', classes=3 , encoder_weights="imagenet", activation=None)
model = model.to(device)

model_path = './statedict/effi2unet_baseline.pt'

ckpt = torch.load(model_path, map_location = device)

try:
    model.load_state_dict(ckpt['state_dict'])
except:
    model.load_state_dict(ckpt)

model.eval()

for step, (images, masks, _) in tqdm(enumerate(val_loader)):

    # images - 1,3,512,512 > tuple

    img = images[0] # 3,512,512
    img = img.unsqueeze(0)

    pred = model(img.to(device)) # 1,3,512,512
    pred = torch.argmax(pred, dim=1).detach().cpu() # .numpy() # (1,3,512,512) > (1,512,512)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 12))

    # 더 뚜렷하게
    img[img > 1] = 1
    img[img <-1] = -1

    ax1.imshow(img[0][0], cmap='gray')
    ax1.grid(False)
    ax1.set_title('img')

    ax2.imshow(masks[0])
    ax2.grid(False)
    unique = np.array(torch.unique(masks[0]),dtype=np.uint8)
    unique = ''.join(str(x) for x in unique)
    ax2.set_title(f'gt_{unique}')

    ax3.imshow(pred[0])
    ax3.grid(False)
    unique = np.unique(pred[0])
    unique = ''.join(str(x) for x in unique)
    ax3.set_title(f'pred_crop_{unique}')

    plt.show()
    plt.savefig(f'./visualize/results_{step}.png')
    plt.close()

    if step > 20:
        break






