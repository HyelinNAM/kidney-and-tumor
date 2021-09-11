import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from dataset import make_foldset
from utils import collate_fn, define_transform, save_model
from metric import case2dice
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu" 

train_dataset, val_dataset = make_foldset(fold=0)
val_loader = DataLoader(dataset= val_dataset, batch_size=4, collate_fn=collate_fn, shuffle=False)

model = smp.Unet(encoder_name='tu-efficientnetv2_rw_m', classes=3 , encoder_weights='imagenet', activation=None) #tu-efficientnetv2_rw_m
model = model.to(device)

model_path = './statedict/effi2unet_baseline.pt'
ckpt = torch.load(model_path, map_location = device)

try:
    model.load_state_dict(ckpt['state_dict'])
except:
    model.load_state_dict(ckpt)

model.eval()

cnt = 0
case = 0 

total_dice = 0
total_kidney = 0
total_tumor = 0

gt_list = []
kidney_list = []
tumor_list = []

for step, (images, masks, _) in tqdm(enumerate(val_loader, start=1)):

    images = torch.stack(images) # bs,3,512,512
    masks = torch.stack(masks).long()

    images, masks = images.to(device), masks 

    outputs = model(images)
    cnt += 1

    max_outputs = torch.argmax(outputs, dim=1).detach().cpu()

    gt_list.append(masks.detach().cpu().numpy())

    base = torch.full((4,512,512),0)

    kidney_where = max_outputs.detach().cpu() == 1 #8,512,512
    kidney = torch.where(kidney_where, max_outputs, base) # 0,1

    tumor_where = max_outputs.detach().cpu() == 2
    tumor = torch.where(tumor_where, max_outputs, base) # 0,2

    kidney_list.append(kidney.numpy())
    tumor_list.append(tumor.numpy())

    if step % 16 == 0:
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

print(f'Kidney {avg_kidney:.3f} Tumor {avg_tumor:.3f}')
print(f'Evaluation Avg dice: {avg_dice:.3f}')