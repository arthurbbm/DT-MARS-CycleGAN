# #!/usr/bin/env python3
# """
# Training-only script for semantic segmentation using HuggingFace Transformers.
# Saves model checkpoints every epoch (or by step if configured).
# Usage:
#     python train.py \
#         --epoch 0 \
#         --n_epochs 10 \
#         --batchSize 4 \
#         --dataroot /path/to/dataset \
#         --outdir /path/to/save/model \
#         --lr 1e-4 \
#         --save_strategy epoch \
#         [--save_steps 100] \
#         [--save_total_limit 2] \
#         [--logging_dir ./logs] \
#         [--logging_steps 50] \
#         [--num_labels 5] \
#         [--device cuda]
# """
# import os
# import argparse
# import numpy as np
# from PIL import Image
# import torch
# from torchvision import transforms
# from transformers import (
#     AutoImageProcessor,
#     AutoModelForSemanticSegmentation,
#     TrainingArguments,
#     Trainer
# )
#
# def get_image_mask_pairs(image_dir, mask_dir):
#     image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
#     mask_files  = sorted([f for f in os.listdir(mask_dir)  if f.lower().endswith(('.png', '.jpg'))])
#     return [
#         {
#             'image': Image.open(os.path.join(image_dir, img)).convert('RGB'),
#             'mask':  Image.open(os.path.join(mask_dir, msk))
#         }
#         for img, msk in zip(image_files, mask_files)
#     ]
#
# class SegDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings
#     def __len__(self):
#         return len(self.encodings)
#     def __getitem__(self, idx):
#         return self.encodings[idx]
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Train segmentation model")
#     parser.add_argument('--epoch', type=int, default=0,
#                         help="starting epoch")
#     parser.add_argument('--n_epochs', type=int, default=10,
#                         help="number of epochs of training")
#     parser.add_argument('--batchSize', type=int, default=4,
#                         help="batch size per device")
#     parser.add_argument('--dataroot', type=str, required=True,
#                         help="root directory of the dataset (expects train/images & train/masks subfolders)")
#     parser.add_argument('--outdir', type=str, default='./fine_tuned_model',
#                         help="where to save model checkpoints")
#     parser.add_argument('--lr', type=float, default=1e-4,
#                         help="learning rate")
#     parser.add_argument('--save_strategy', choices=['no','epoch','steps'], default='epoch',
#                         help="when to save model: 'epoch' or 'steps'")
#     parser.add_argument('--save_steps', type=int, default=None,
#                         help="number of steps between saves (if save_strategy='steps')")
#     parser.add_argument('--save_total_limit', type=int, default=2,
#                         help="max number of checkpoint folders to keep")
#     parser.add_argument('--logging_dir', default='./logs',
#                         help="TensorBoard logging directory")
#     parser.add_argument('--logging_steps', type=int, default=50,
#                         help="logging frequency in steps")
#     parser.add_argument('--num_labels', type=int, default=5,
#                         help="number of segmentation labels/classes")
#     parser.add_argument('--device', default=None, choices=['cpu','cuda','mps'],
#                         help="compute device (overrides auto-detect)")
#     args = parser.parse_args()
#
#     # Paths
#     image_dir = os.path.join(args.dataroot, 'train', 'images')
#     mask_dir  = os.path.join(args.dataroot, 'train', 'masks')
#
#     # Device selection
#     if args.device:
#         device = torch.device(args.device)
#     else:
#         if torch.cuda.is_available():
#             device = torch.device('cuda')
#         elif getattr(torch.backends, 'mps', False) and torch.backends.mps.is_available():
#             device = torch.device('mps')
#         else:
#             device = torch.device('cpu')
#     print(f"Using device: {device}")
#
#     # Data augmentations
#     train_transforms = transforms.Compose([
#         transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
#         transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5)
#     ])
#
#     # Model & processor
#     pretrained = 'google/deeplabv3_mobilenet_v2_1.0_513'
#     processor = AutoImageProcessor.from_pretrained(pretrained)
#     model = AutoModelForSemanticSegmentation.from_pretrained(
#         pretrained,
#         ignore_mismatched_sizes=True,
#         num_labels=args.num_labels
#     ).to(device)
#
#     # Prepare encodings
#     examples = get_image_mask_pairs(image_dir, mask_dir)
#     encodings = []
#     for ex in examples:
#         aug_img = train_transforms(ex['image'])
#         enc = processor(images=aug_img, return_tensors='pt')
#         pixel_values = enc['pixel_values'].squeeze(0).to(device)
#         _, H, W = pixel_values.shape
#         mask_resized = ex['mask'].resize((W, H), resample=Image.NEAREST)
#         labels = torch.tensor(np.array(mask_resized), dtype=torch.long)
#         encodings.append({'pixel_values': pixel_values, 'labels': labels})
#
#     # Split dataset
#     split = int(0.8 * len(encodings))
#     train_ds = SegDataset(encodings[:split])
#     eval_ds  = SegDataset(encodings[split:])
#
#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir=args.outdir,
#         num_train_epochs=args.n_epochs,
#         per_device_train_batch_size=args.batchSize,
#         learning_rate=args.lr,
#         save_strategy=args.save_strategy,
#         save_steps=args.save_steps,
#         save_total_limit=args.save_total_limit,
#         logging_dir=args.logging_dir,
#         logging_steps=args.logging_steps,
#         do_train=True,
#         do_eval=False
#     )
#
#     # Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         eval_dataset=eval_ds,
#         tokenizer=processor
#     )
#
#     # Train and save
#     trainer.train(resume_from_checkpoint=None)
#     trainer.save_model(args.outdir)
#     processor.save_pretrained(args.outdir)
#     print(f"Training complete. Checkpoints in {args.outdir}")
#
# if __name__ == '__main__':
#     main()



















#!/usr/bin/env python3
"""
Training-only script for semantic segmentation using HuggingFace Transformers,
but saving a single `segmentation_model.pth` state_dict at the end.
Usage:
    python train.py \
        --n_epochs 10 \
        --batchSize 4 \
        --dataroot /path/to/dataset \
        --outdir /path/to/save/model \
        --lr 1e-4 \
        --save_strategy epoch \
        --num_labels 5 \
        [--device cuda]
"""
import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    TrainingArguments,
    Trainer
)

def get_image_mask_pairs(image_dir, mask_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
    mask_files  = sorted([f for f in os.listdir(mask_dir)  if f.lower().endswith(('.png', '.jpg'))])
    return [
        {
            'image': Image.open(os.path.join(image_dir, img)).convert('RGB'),
            'mask':  Image.open(os.path.join(mask_dir, msk))
        }
        for img, msk in zip(image_files, mask_files)
    ]

class SegDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings)
    def __getitem__(self, idx):
        return self.encodings[idx]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--batchSize', type=int, default=4)
    p.add_argument('--dataroot',  type=str, required=True)
    p.add_argument('--outdir',    type=str, default='./fine_tuned_model')
    p.add_argument('--lr',        type=float, default=1e-4)
    p.add_argument('--save_strategy', choices=['no','epoch','steps'], default='epoch')
    p.add_argument('--save_steps', type=int, default=None)
    p.add_argument('--save_total_limit', type=int, default=2)
    p.add_argument('--logging_dir', default='./logs')
    p.add_argument('--logging_steps', type=int, default=50)
    p.add_argument('--num_labels', type=int, default=5)
    p.add_argument('--device', default=None, choices=['cpu','cuda','mps'])
    return p.parse_args()

def main():
    args = parse_args()

    # device
    if args.device:
        device = torch.device(args.device)
    else:
        device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('mps') if getattr(torch.backends, 'mps', False) and torch.backends.mps.is_available()
            else torch.device('cpu')
        )
    print(f"Using device: {device}")

    # data dirs
    image_dir = os.path.join(args.dataroot, 'images')
    mask_dir  = os.path.join(args.dataroot, 'masks')

    # augmentations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5)
    ])

    # model + processor
    pretrained = 'google/deeplabv3_mobilenet_v2_1.0_513'
    processor = AutoImageProcessor.from_pretrained(pretrained)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        pretrained,
        ignore_mismatched_sizes=True,
        num_labels=args.num_labels
    ).to(device)

    # prepare examples
    examples = get_image_mask_pairs(image_dir, mask_dir)
    encodings = []
    for ex in examples:
        aug_img = train_transforms(ex['image'])
        enc = processor(images=aug_img, return_tensors='pt')
        pixel_values = enc['pixel_values'].squeeze(0).to(device)
        _, H, W = pixel_values.shape
        mask_rs = ex['mask'].resize((W, H), resample=Image.NEAREST)
        labels = torch.tensor(np.array(mask_rs), dtype=torch.long).to(device)
        encodings.append({'pixel_values': pixel_values, 'labels': labels})

    # split
    split = int(0.8 * len(encodings))
    train_ds = SegDataset(encodings[:split])
    eval_ds  = SegDataset(encodings[split:])

    # trainer args
    training_args = TrainingArguments(
        output_dir=args.outdir,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batchSize,
        learning_rate=args.lr,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        do_train=True,
        do_eval=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor  # HF will use this to collate / log
    )

    # train
    trainer.train()
    # instead of HF's `save_model`, dump a single .pth
    os.makedirs(args.outdir, exist_ok=True)
    pth_path = os.path.join(args.outdir, 'segmentation_model.pth')
    torch.save(model.state_dict(), pth_path)
    print(f"✅ Saved state_dict to {pth_path}")

if __name__ == '__main__':
    main()


















# #!/usr/bin/env python3
# """
# Plain PyTorch training loop for Qualcomm DeepLabV3‑ResNet50 + edge‑aware UNet decoder.
#
# UNet decoder now concatenates skip features to avoid channel mismatches.
#
# Usage:
#     python train.py \
#       --n_epochs 10 \
#       --batchSize 4 \
#       --dataroot /path/to/segmentation/dataset \
#       --outdir ./checkpoints \
#       --lr 5e-5 \
#       --num_labels 5 \
#       --device cuda
# """
# import os
# import argparse
# import numpy as np
# from PIL import Image
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# import cv2
# from qai_hub_models.models.deeplabv3_resnet50 import Model as QaiModel
#
# # UNet-style decoder with concatenation (fixed channel flow)
# class UNetDecoder(nn.Module):
#     def __init__(self, encoder_channels, decoder_channels):
#         super().__init__()
#         # encoder_channels: [c5, c4, c3, c2]
#         # decoder_channels length must be len(encoder_channels) - 1
#         assert len(decoder_channels) == len(encoder_channels) - 1, (
#             "decoder_channels must be one less than encoder_channels"
#         )
#
#         layers = []
#         in_ch = encoder_channels[0]
#         # iterate over skip levels and desired output channels
#         for skip_ch, out_ch in zip(encoder_channels[1:], decoder_channels):
#             layers.append(nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#                 nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#             ))
#             in_ch = out_ch  # next block input channels = current out_ch
#
#         self.blocks = nn.ModuleList(layers)
#
#     def forward(self, feats):
#         # feats: [c5, c4, c3, c2]
#         x = feats[0]
#         for block, skip in zip(self.blocks, feats[1:]):
#             x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
#             x = torch.cat([x, skip], dim=1)
#             x = block(x)
#         return x
#
# # Full model: DeepLabV3 backbone + UNet decoder + edge head
# class EdgeSegModel(nn.Module):
#     def __init__(self, num_labels):
#         super().__init__()
#         # Load QAI DeepLabV3-ResNet50
#         wrapper = QaiModel.from_pretrained()
#         self.seg_model = wrapper.model  # torchvision DeepLabV3
#
#         # Replace final classifier conv
#         old_conv = self.seg_model.classifier[-1]
#         self.seg_model.classifier[-1] = nn.Conv2d(
#             old_conv.in_channels, num_labels, kernel_size=1
#         )
#
#         # Build decoder
#         enc_ch = [2048, 1024, 512, 256]
#         dec_ch = [256, 128, 64]  # one block per skip (c4, c3, c2)
#         self.decoder = UNetDecoder(enc_ch, dec_ch)
#         self.dec_head = nn.Conv2d(dec_ch[-1], num_labels, kernel_size=1)
#
#         # Edge detection head from c3
#         self.edge_head = nn.Conv2d(512, 1, kernel_size=1)
#
#     def forward(self, x):
#         # 1) DeepLab segmentation output
#         seg_out = self.seg_model(x)['out']
#
#         # 2) Extract ResNet backbone features
#         body = self.seg_model.backbone
#         c1 = body.relu(body.bn1(body.conv1(x)))
#         c1 = body.maxpool(c1)
#         c2 = body.layer1(c1)
#         c3 = body.layer2(c2)
#         c4 = body.layer3(c3)
#         c5 = body.layer4(c4)
#
#         # 3) UNet decode
#         feats = [c5, c4, c3, c2]
#         x_dec = self.decoder(feats)
#         dec_logits = self.dec_head(x_dec)
#         dec_logits = F.interpolate(
#             dec_logits, size=x.shape[-2:], mode='bilinear', align_corners=False
#         )
#
#         # 4) Edge head
#         edge_logits = self.edge_head(c3)
#         edge_logits = F.interpolate(
#             edge_logits, size=x.shape[-2:], mode='bilinear', align_corners=False
#         )
#
#         # Combine segmentation heads
#         seg_logits = seg_out + dec_logits
#         return seg_logits, edge_logits
#
# # Combined loss: CE + Dice for seg, BCE for edges
# class CombinedLoss(nn.Module):
#     def __init__(self, dice_w=1.0, edge_w=1.0):
#         super().__init__()
#         self.ce = nn.CrossEntropyLoss()
#         self.dice_w = dice_w
#         self.bce    = nn.BCEWithLogitsLoss()
#         self.edge_w = edge_w
#
#     def dice_loss(self, logits, labels, eps=1e-6):
#         probs = torch.softmax(logits, dim=1)
#         B, C, H, W = probs.shape
#         onehot = F.one_hot(labels, C).permute(0,3,1,2).float()
#         loss = 0.0
#         for c in range(C):
#             p = probs[:,c]
#             t = onehot[:,c]
#             inter = (p * t).sum([1,2])
#             union = p.sum([1,2]) + t.sum([1,2])
#             loss += 1 - (2*inter + eps) / (union + eps)
#         return loss.mean()
#
#     def forward(self, seg_logits, labels, edge_logits, edge_tgt):
#         loss_seg  = self.ce(seg_logits, labels) + self.dice_w * self.dice_loss(seg_logits, labels)
#         loss_edge = self.bce(edge_logits, edge_tgt)
#         return loss_seg + self.edge_w * loss_edge
#
# # Dataset delivering (img, mask, edge) triples
# class SegDataset(Dataset):
#     def __init__(self, img_dir, msk_dir, size=513):
#         self.img_paths = sorted(
#             os.path.join(img_dir, f)
#             for f in os.listdir(img_dir)
#             if f.lower().endswith(('.jpg','.png'))
#         )
#         self.msk_paths = sorted(
#             os.path.join(msk_dir, f)
#             for f in os.listdir(msk_dir)
#             if f.lower().endswith(('.png','.jpg'))
#         )
#         self.size = size
#         self.tf = transforms.Compose([
#             transforms.RandomResizedCrop(size, scale=(0.5,1.0)),
#             transforms.RandomHorizontalFlip(0.5),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
#         ])
#
#     def __len__(self):
#         return len(self.img_paths)
#
#     def __getitem__(self, idx):
#         img = Image.open(self.img_paths[idx]).convert('RGB')
#         msk = Image.open(self.msk_paths[idx]).convert('L')
#         img_t = self.tf(img)
#         lbl = torch.tensor(
#             np.array(msk.resize((self.size,self.size), Image.NEAREST)),
#             dtype=torch.long
#         )
#         # generate binary edge map
#         mask_np = (lbl.numpy() > 0).astype(np.uint8) * 255
#         edge_np = cv2.Canny(mask_np,100,200) / 255.0
#         edge_t = torch.tensor(edge_np, dtype=torch.float32).unsqueeze(0)
#         return img_t, lbl, edge_t
#
# # Train one epoch
# def train_epoch(model, loader, optimizer, loss_fn, device):
#     model.train()
#     total_loss = 0.0
#     for imgs, labels, edges in loader:
#         imgs, labels, edges = imgs.to(device), labels.to(device), edges.to(device)
#         optimizer.zero_grad()
#         seg_logits, edge_logits = model(imgs)
#         loss = loss_fn(seg_logits, labels, edge_logits, edges)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(loader)
#
# # Main
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--n_epochs',  type=int, default=10)
#     parser.add_argument('--batchSize', type=int, default=4)
#     parser.add_argument('--dataroot',  type=str, required=True)
#     parser.add_argument('--outdir',    type=str, default='./checkpoints')
#     parser.add_argument('--lr',        type=float, default=5e-5)
#     parser.add_argument('--num_labels',type=int, default=5)
#     parser.add_argument('--device',    choices=['cpu','cuda','mps'], default=None)
#     args = parser.parse_args()
#
#     device = torch.device(args.device) if args.device else (
#         torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     )
#     print(f"Using device: {device}")
#
#     model     = EdgeSegModel(args.num_labels).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     loss_fn   = CombinedLoss()
#
#     train_img_dir = os.path.join(args.dataroot, 'images')
#     train_msk_dir = os.path.join(args.dataroot, 'masks')
#     dataset = SegDataset(train_img_dir, train_msk_dir, size=513)
#     loader  = DataLoader(dataset, batch_size=args.batchSize, shuffle=True,
#                          num_workers=4, pin_memory=True)
#
#     os.makedirs(args.outdir, exist_ok=True)
#     for epoch in range(1, args.n_epochs+1):
#         avg_loss = train_epoch(model, loader, optimizer, loss_fn, device)
#         print(f"Epoch {epoch}/{args.n_epochs} — loss: {avg_loss:.4f}")
#         ckpt = os.path.join(args.outdir,'checkpoint.pth')
#         torch.save(model.state_dict(), ckpt)
#
#     seg_model = os.path.join(args.outdir,'segmentation_model.pth')
#     os.rename(ckpt, seg_model)

