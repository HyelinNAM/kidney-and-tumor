import os
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import make_foldset
from utils import set_seed, collate_fn, define_transform, save_model, define_scheduler, load_ckpt
from metric import case2dice
from tqdm import tqdm
from loss import define_loss

import wandb

def train(model, num_epochs, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, val_every, device, args):

    print('Start training...')
    best_loss = 0

    if args.wandb:
        wandb.watch(model, criterion, log='all', log_freq= 10)

    example_ct = 0

    for epoch in range(num_epochs):

        model.train()

        for step, (images, masks, _) in enumerate(tqdm(train_loader)):
            images = torch.stack(images) # b,c,h,w
            masks = torch.stack(masks).long() # b,c,h,w

            images, masks = images.to(device), masks.to(device)

            outputs = model(images) # 8,3,512,512

            if 'combo' in args.loss:

                if args.loss == 'combo1':

                    bce = criterion[0]
                    dice = criterion[1]

                    loss = 0.75 * bce(outputs, masks) + 0.25 * dice(outputs, masks)

                elif args.loss == 'combo2':

                    focal = criterion[0]
                    dice = criterion[1]

                    loss = 0.75 * focal(outputs, masks) + 0.25 * dice(outputs, masks)

                elif args.loss == 'combo3':
                    
                    focal = criterion[0]
                    dice = criterion[1]

                    loss = 0.5 * focal(outputs, masks) + 0.5 * dice(outputs, masks)

                elif args.loss == 'combo4':

                    focal = criterion[0]
                    dice = criterion[1]

                    loss = 0.25 * focal(outputs, masks) + 0.75 * dice(outputs, masks)
                
                else:
                    loss = criterion(outputs, masks)

            
            example_ct += len(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if 'cosine' in args.lr_scheduler:
                scheduler.step() # cosine scehduler는 iter마다 step

            if args.wandb and (step+1) % 25 == 0:
                wandb.log({"epoch": epoch, "loss": loss, "lr":optimizer.param_groups[0]["lr"]}, step=example_ct)
            
            if (step+1) % 300 == 0:
                print(f'Epoch {epoch}/{num_epochs}, Step {step}/{len(train_loader)}, Loss {loss.item():.3f}')

        if (epoch+1) % val_every == 0:
            avg_dice = validation(model, epoch+1, val_loader, criterion, device, args)
            
        # epoch마다 step하는 scheduler
        if 'plateau' in args.lr_scheduler:
            scheduler.step(avg_dice)

        if 'step' in args.lr_scheduler:
            scheduler.step()

        print(f'### lr : {optimizer.param_groups[0]["lr"]}')   

        if avg_dice >= best_loss:
            print(f'Best performance at epoch {epoch+1}!')
            print(f'Save model in {saved_dir} ...')
            
            best_loss = avg_dice

            checkpoint = {
                'epoch':epoch +1,
                'state_dict': model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict()
            }

            save_model(checkpoint, saved_dir, file_name = f'{args.save}.pt')

def validation(model, epoch, val_loader, device, args):
    print('Start validation...')

    model.eval()

    with torch.no_grad():
        total_loss = 0
        cnt = 0
        case = 0

        total_dice = 0
        total_kidney = 0
        total_tumor = 0

        gt_list = []
        kidney_list = []
        tumor_list = []

        for step, (images, masks, _) in enumerate(tqdm(val_loader) ,start=1): # len(val_loader) = 200 

            images = torch.stack(images)
            masks = torch.stack(masks).long()

            images = images.to(device)

            outputs = model(images) # (8,3,512,512)

            cnt += 1

            max_outputs = torch.argmax(outputs, dim=1).detach().cpu() # (8,512,512)

            base = torch.full((4,512,512),0)

            gt_list.append(masks.detach().cpu().numpy())
            
            kidney_where = max_outputs.detach().cpu() == 1
            kidney = torch.where(kidney_where, max_outputs, base)

            tumor_where = max_outputs.detach().cpu() == 2
            tumor = torch.where(tumor_where, max_outputs, base)

            kidney_list.append(kidney.numpy())
            tumor_list.append(tumor.numpy())

            if step % 16 == 0: # bs-4
                kidney, tumor, dice = case2dice(gt_list, kidney_list, tumor_list)

                total_dice += dice
                total_kidney += kidney
                total_tumor += tumor

                case += 1

                gt_list = []
                kidney_list = []
                tumor_list = []

        avg_kidney = total_kidney / case
        avg_tumor = total_tumor / case
        avg_dice = total_dice / case

        print(f'Validation #{epoch} Avg dice: {avg_dice:.3f} Kidney {avg_kidney:.3f} Tumor {avg_tumor:.3f}')

        if args.wandb:
            wandb.log({'Val dice': avg_dice})

    return avg_dice

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['xl', 'l', 'm'])
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--new_norm', action='store_true')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--loss', type=str, choices=['bce','focal','lovasz', 'dice', 'combo1', 'combo2', 'combo3', 'combo4'])
    parser.add_argument('--lr_scheduler', type=str, choices=['plateau', 'cosine', 'step', 'nope'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optim', type=str, choices=['adam','sgd','adamw'])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save',type=str, required=True)
    args = parser.parse_args()

    print(args)

    # seed 고정
    set_seed(random_seed=28)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyperparameter
    config = dict(
        num_epochs=args.epoch,
        classes=3,
        batch_size=4,
        learning_rate=args.lr,
        loss = args.loss,
        lr_scheduler = args.lr_scheduler,
        save = args.save
    )

    with wandb.init(project='unet_effiv2', name=args.save, config=config, entity='hateseg'):

        config = wandb.config

        train_dataset, val_dataset = make_foldset(fold=args.fold, crop=args.crop, new_norm = args.new_norm)

        train_loader = DataLoader(dataset= train_dataset,batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)
        val_loader = DataLoader(dataset= val_dataset,batch_size=config.batch_size, collate_fn=collate_fn)

        model = smp.Unet(encoder_name='tu-efficientnetv2_rw_m', classes=3 , encoder_weights="imagenet", activation=None)
        # xl - encoder_name='tu-tf_efficientnetv2_xl_in21ft1k' / l - encoder_name='tu-tf_efficientnetv2_l_in21ft1k'

        model = model.to(device)

        criterion = define_loss(args.loss)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate, weight_decay=1e-6)

        total = len(train_dataset) * config.num_epochs
        scheduler = define_scheduler(args.lr_scheduler, optimizer, total, config.learning_rate)

        train(model, config.num_epochs, train_loader, val_loader, criterion, optimizer, scheduler,
        saved_dir='./statedict', val_every= 1, device = device, args=args)