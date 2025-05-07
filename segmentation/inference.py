#!/usr/bin/env python3
"""
Inference script for semantic segmentation using a trained DeepLabV3-ResNet50 model.

Usage:
    python inference.py \
        --model_path /path/to/segmentation_model.pth \
        --num_labels 5 \
        --test_dir /path/to/test/images \
        --output_dir /path/to/save/predicted/masks \
        [--device cuda]
"""
import os
import argparse
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from PIL import Image
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Semantic segmentation inference")
    p.add_argument('--model_path', type=str, required=True,
                   help='Path to the trained segmentation_model.pth')
    p.add_argument('--num_labels', type=int, default=5,
                   help='Number of segmentation classes')
    p.add_argument('--test_dir', type=str, required=True,
                   help='Directory with input images')
    p.add_argument('--output_dir', type=str, default='./inference_results',
                   help='Directory to save output masks')
    p.add_argument('--device', choices=['cpu','cuda','mps'], default=None,
                   help='Compute device')
    return p.parse_args()

def load_model(model_path, num_labels, device):
    # recreate model architecture
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
        progress=True,
        num_classes=num_labels
    )
    model.classifier = DeepLabHead(2048, num_labels)
    # load state dict
    state = torch.load(model_path, map_location=device)
    # if full checkpoint dict, extract state_dict
    if isinstance(state, dict) and 'model_state_dict' in state:
        state_dict = state['model_state_dict']
    else:
        state_dict = state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_transform():
    # only normalization, no augmentation
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])


def predict_single(model, img_tensor, device):
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)['out']  # (1, C, H, W)
        pred = torch.argmax(out.squeeze(), dim=0).cpu().numpy().astype(np.uint8)
    return pred


def main():
    args = parse_args()

    # device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load model
    model = load_model(args.model_path, args.num_labels, device)

    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare transform
    transform = get_transform()

    # iterate over test images
    for fname in sorted(os.listdir(args.test_dir)):
        if not fname.lower().endswith(('.jpg','jpeg','.png')):
            continue
        img_path = os.path.join(args.test_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        mask_np = predict_single(model, img_tensor, device)

        # save mask as PNG
        mask_img = Image.fromarray(mask_np)
        out_path = os.path.join(args.output_dir, os.path.splitext(fname)[0] + '_mask.png')
        mask_img.save(out_path)
        print(f"Saved mask: {out_path}")

if __name__ == '__main__':
    main()
