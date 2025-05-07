# #!/usr/bin/env python3
# """
# Inference script for semantic segmentation using a model trained with EdgeSegModel.
# Usage:
#     python inference.py \
#         --model_path path/to/segmentation_model.pth \
#         --num_labels 5 \
#         --test_dir  path/to/test/images \
#         --output_dir path/to/save/masks \
#         [--size 513] \
#         [--device {cpu,cuda,mps}]
# """
# import os
# import argparse
# import numpy as np
# from PIL import Image
# import torch
# import torch.nn.functional as F
# from torch import nn
# from torchvision import transforms
#
# # --- copy these from your training script ---
# class UNetDecoder(nn.Module):
#     def __init__(self, encoder_channels, decoder_channels):
#         super().__init__()
#         assert len(decoder_channels) == len(encoder_channels) - 1
#         layers = []
#         in_ch = encoder_channels[0]
#         for skip_ch, out_ch in zip(encoder_channels[1:], decoder_channels):
#             layers.append(nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#                 nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#             ))
#             in_ch = out_ch
#         self.blocks = nn.ModuleList(layers)
#
#     def forward(self, feats):
#         x = feats[0]
#         for block, skip in zip(self.blocks, feats[1:]):
#             x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
#             x = torch.cat([x, skip], dim=1)
#             x = block(x)
#         return x
#
# class EdgeSegModel(nn.Module):
#     def __init__(self, num_labels):
#         super().__init__()
#         from qai_hub_models.models.deeplabv3_resnet50 import Model as QaiModel
#         wrapper = QaiModel.from_pretrained()
#         self.seg_model = wrapper.model  # torchvision DeepLabV3 backbone+head
#
#         # replace classifier to match num_labels
#         old = self.seg_model.classifier[-1]
#         self.seg_model.classifier[-1] = nn.Conv2d(old.in_channels, num_labels, kernel_size=1)
#
#         # UNet decoder on backbone features
#         enc_ch = [2048, 1024, 512, 256]
#         dec_ch = [256, 128, 64]
#         self.decoder = UNetDecoder(enc_ch, dec_ch)
#         self.dec_head = nn.Conv2d(dec_ch[-1], num_labels, kernel_size=1)
#
#         # edge head (unused in this script)
#         self.edge_head = nn.Conv2d(512, 1, kernel_size=1)
#
#     def forward(self, x):
#         seg_out = self.seg_model(x)['out']
#         body = self.seg_model.backbone
#         c1 = body.relu(body.bn1(body.conv1(x)))
#         c1 = body.maxpool(c1)
#         c2 = body.layer1(c1)
#         c3 = body.layer2(c2)
#         c4 = body.layer3(c3)
#         c5 = body.layer4(c4)
#
#         feats = [c5, c4, c3, c2]
#         x_dec = self.decoder(feats)
#         dec_logits = self.dec_head(x_dec)
#         dec_logits = F.interpolate(dec_logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
#
#         # combine
#         seg_logits = seg_out + dec_logits
#         # edge_logits = ...
#         return seg_logits, None
# # --- end copied classes ---
#
# def parse_args():
#     parser = argparse.ArgumentParser(description="Semantic segmentation inference script")
#     parser.add_argument('--model_path', required=True,
#                         help="Path to the .pth file containing your trained model.state_dict()")
#     parser.add_argument('--num_labels', type=int, required=True,
#                         help="Number of segmentation classes used in training")
#     parser.add_argument('--test_dir', required=True,
#                         help="Directory with test images (*.jpg, *.png)")
#     parser.add_argument('--output_dir', required=True,
#                         help="Where to save predicted mask images")
#     parser.add_argument('--size', type=int, default=513,
#                         help="Resize shorter edge to this for inference (default: 513)")
#     parser.add_argument('--device', choices=['cpu','cuda','mps'], default=None,
#                         help="Compute device. Default auto-detects")
#     return parser.parse_args()
#
# def main():
#     args = parse_args()
#     os.makedirs(args.output_dir, exist_ok=True)
#
#     # device
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
#     # load model
#     model = EdgeSegModel(args.num_labels)
#     state_dict = torch.load(args.model_path, map_location=device)
#     model.load_state_dict(state_dict)
#     model.to(device).eval()
#
#     # preprocess
#     tf = transforms.Compose([
#         transforms.Resize(args.size, interpolation=Image.BILINEAR),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
#     ])
#
#     # inference
#     for fname in sorted(os.listdir(args.test_dir)):
#         if not fname.lower().endswith(('.jpg','.png')):
#             continue
#         img_path = os.path.join(args.test_dir, fname)
#         img = Image.open(img_path).convert('RGB')
#         orig_w, orig_h = img.size
#
#         x = tf(img).unsqueeze(0).to(device)
#         with torch.no_grad():
#             seg_logits, _ = model(x)
#             # upsample to original size
#             seg_logits = F.interpolate(seg_logits, size=(orig_h, orig_w),
#                                        mode='bilinear', align_corners=False)
#             mask = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
#
#         mask_img = Image.fromarray(mask, mode='L')
#         out_name = os.path.splitext(fname)[0] + '_mask.png'
#         mask_img.save(os.path.join(args.output_dir, out_name))
#         print(f"Saved mask: {out_name}")
#
# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
"""
Inference script for semantic segmentation using a model trained with
HuggingFace’s AutoModelForSemanticSegmentation (deeplabv3_mobilenet_v2).
Loads a single `state_dict` checkpoint saved as `segmentation_model.pth`.
Usage:
    python inference.py \
      --pretrained_model google/deeplabv3_mobilenet_v2_1.0_513 \
      --model_path /path/to/segmentation_model.pth \
      --num_labels 5 \
      --test_dir /path/to/images \
      --output_dir /path/to/masks \
      [--device cuda]
"""
import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pretrained_model", type=str,
        default="google/deeplabv3_mobilenet_v2_1.0_513",
        help="HuggingFace model ID (architecture & default weights)"
    )
    p.add_argument(
        "--model_path", required=True,
        help="Path to the .pth state_dict saved at training end"
    )
    p.add_argument(
        "--num_labels", type=int, required=True,
        help="Number of classes your model was trained on"
    )
    p.add_argument(
        "--test_dir", required=True,
        help="Directory with test images (*.jpg, *.png)"
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Where to save predicted mask images"
    )
    p.add_argument(
        "--device", choices=["cpu","cuda","mps"], default=None,
        help="Compute device (overrides auto-detect)"
    )
    return p.parse_args()

def main():
    args = parse_args()
    # device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", False) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # load processor + architecture
    processor = AutoImageProcessor.from_pretrained(args.pretrained_model)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        args.pretrained_model,
        ignore_mismatched_sizes=True,
        num_labels=args.num_labels
    )
    # load our fine‑tuned weights
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device).eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # iterate images
    for fname in sorted(os.listdir(args.test_dir)):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        img_path = os.path.join(args.test_dir, fname)
        image = Image.open(img_path).convert("RGB")
        # preprocess
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            seg = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0]

        mask_np = seg.cpu().numpy().astype(np.uint8)
        mask_img = Image.fromarray(mask_np, mode="L")
        out_name = os.path.splitext(fname)[0] + "_mask.png"
        mask_img.save(os.path.join(args.output_dir, out_name))
        print(f"Saved mask: {out_name}")

if __name__ == "__main__":
    main()





# #!/usr/bin/env python3
# """
# Inference script for semantic segmentation using a fine-tuned Transformers model.
# Usage:
#     python inference.py \
#         --model_dir path/to/fine_tuned_model \
#         --test_dir  path/to/test/images \
#         --output_dir path/to/save/masks \
#         [--device {cpu,cuda,mps}]
# """
# import os
# import argparse
# import numpy as np
# from PIL import Image
# import torch
# from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
#
# def parse_args():
#     parser = argparse.ArgumentParser(description="Semantic segmentation inference script")
#     parser.add_argument('--model_dir', required=True,
#                         help="Path to the directory containing the fine-tuned model and processor")
#     parser.add_argument('--test_dir', required=True,
#                         help="Directory with test images (*.jpg, *.png)")
#     parser.add_argument('--output_dir', required=True,
#                         help="Where to save predicted mask images")
#     parser.add_argument('--device', default=None, choices=['cpu','cuda','mps'],
#                         help="Compute device. Default auto-detects")
#     return parser.parse_args()
#
#
# def main():
#     args = parse_args()
#     os.makedirs(args.output_dir, exist_ok=True)
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
#
#     print(f"Using device: {device}")
#
#     # Load processor and model
#     processor = AutoImageProcessor.from_pretrained(args.model_dir)
#     model = AutoModelForSemanticSegmentation.from_pretrained(
#         args.model_dir
#     ).to(device)
#     model.eval()
#
#     # Inference over test images
#     for fname in sorted(os.listdir(args.test_dir)):
#         if not fname.lower().endswith(('.jpg', '.png')):
#             continue
#         img_path = os.path.join(args.test_dir, fname)
#         image = Image.open(img_path).convert('RGB')
#
#         # Preprocess
#         enc = processor(images=image, return_tensors='pt')
#         pixel_values = enc['pixel_values'].to(device)
#
#         # Forward
#         with torch.no_grad():
#             outputs = model(pixel_values=pixel_values)
#             seg = processor.post_process_semantic_segmentation(
#                 outputs, target_sizes=[image.size[::-1]]
#             )[0]
#
#         # Convert to mask and save
#         mask_np = seg.cpu().numpy().astype(np.uint8)
#         mask_img = Image.fromarray(mask_np, mode='L')
#         out_name = os.path.splitext(fname)[0] + '_mask.png'
#         mask_img.save(os.path.join(args.output_dir, out_name))
#         print(f"Saved mask: {out_name}")
#
# if __name__ == '__main__':
#     main()


# python inference.py \
#   --model_dir   ./fine_tuned_model \
#   --test_dir    /Users/arthur/Documents/ComputerVision/assignments/DT-MARS-CycleGAN/dataset/segmentation/train \
#   --output_dir  /Users/arthur/Documents/ComputerVision/assignments/DT-MARS-CycleGAN/dataset/segmentation/test_predictions

# python inference.py \
#     --model_dir /home/abhhn/DT-MARS-CycleGAN/segmentation/fine_tuned_model \
#     --test_dir /home/abhhn/data/DT-MARS-CycleGAN/dataset/test \
#     --output_dir /home/abhhn/data/DT-MARS-CycleGAN/dataset/test_predictions \
#     --device cuda

# python inference.py \
#     --model_dir /home/abhhn/DT-MARS-CycleGAN/segmentation/fine_tuned_model \
#     --test_dir /home/abhhn/data/DT-MARS-CycleGAN/dataset/test \
#     --output_dir /home/abhhn/data/DT-MARS-CycleGAN/dataset/test_predictions \
#     --device cuda
