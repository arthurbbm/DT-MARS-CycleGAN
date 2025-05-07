#!/usr/bin/env python3
"""
Training-only script for semantic segmentation using torchvision’s
DeepLabV3-ResNet50, but saving a single `segmentation_model.pth` at the end.

Usage:
    python train.py \
        --n_epochs 10 \
        --batchSize 4 \
        --dataroot /path/to/dataset \
        --outdir /path/to/save/model \
        --checkpoint_freq 5 \
        --lr 1e-4 \
        --num_labels 5 \
        [--device cuda]
"""
import os
import argparse
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--batchSize', type=int, default=4)
    p.add_argument('--dataroot',  type=str, required=True)
    p.add_argument('--outdir',    type=str, default='./fine_tuned_model')
    p.add_argument('--checkpoint_freq', type=int, default=5)
    p.add_argument('--lr',        type=float, default=1e-4)
    p.add_argument('--num_labels', type=int, default=5)
    p.add_argument('--device', choices=['cpu','cuda','mps'], default=None)
    return p.parse_args()

class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.png'))])
        self.masks  = sorted([f for f in os.listdir(mask_dir)  if f.lower().endswith(('.png','.jpg'))])
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.images[idx])).convert('RGB')
        msk = Image.open(os.path.join(self.mask_dir,  self.masks[idx]))
        # apply augmentations + to-tensor + normalize
        img = self.transform(img)
        # masks come in as PIL with values [0, num_classes-1]
        label = torch.from_numpy(np.array(msk)).long()
        return img, label

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs   = torch.stack(imgs)
    labels = torch.stack(labels)
    return imgs, labels

def main():
    args = parse_args()

    # device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    # dirs
    image_dir = os.path.join(args.dataroot, 'images')
    mask_dir  = os.path.join(args.dataroot, 'masks')

    # transforms: augment → to_tensor → normalize (ImageNet stats)
    train_transform = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225]),
    ])

    # dataset + loaders
    full_ds = SegDataset(image_dir, mask_dir, train_transform)
    split = int(0.8 * len(full_ds))
    train_ds, eval_ds = torch.utils.data.random_split(full_ds, [split, len(full_ds)-split])
    train_loader = DataLoader(train_ds, batch_size=args.batchSize, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.batchSize, shuffle=False,
                              collate_fn=collate_fn, pin_memory=True)

    # model: torchvision DeepLabV3 with ResNet50 backbone
    model = deeplabv3_resnet50(weights=None,
                               weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
                               progress=True, num_classes=args.num_labels)

    model.classifier = DeepLabHead(2048, args.num_labels)

    # freeze backbone
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
    model.backbone.eval()  # freeze BN stats
    model.to(device)

    # loss & optimizer (only head params)
    criterion = nn.CrossEntropyLoss()
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(head_params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # training loop
    for epoch in range(1, args.n_epochs+1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)['out']  # (B, C, H, W)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:2d}/{args.n_epochs}  train loss: {avg_loss:.4f}", flush=True)

        if epoch % args.checkpoint_freq == 0:
            # save checkpoint
            os.makedirs(args.outdir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, os.path.join(args.outdir, 'checkpoint.pth'))
            print(f"🔖 Saved checkpoint of epoch {epoch}/{args.n_epochs} to checkpoint.pth", flush=True)

    # save final state_dict
    os.makedirs(args.outdir, exist_ok=True)
    pth_path = os.path.join(args.outdir, 'segmentation_model.pth')
    torch.save(model.state_dict(), pth_path)
    print(f"✅ Saved state_dict to {pth_path}", flush=True)

if __name__ == '__main__':
    main()
