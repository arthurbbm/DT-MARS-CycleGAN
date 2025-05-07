#!/usr/bin/env python3
"""
Inference script for semantic segmentation using a fine-tuned Transformers model.
Usage:
    python inference.py \
        --model_name_or_dir facebook/detr-resnet-50-panoptic \
        --model_path path/to/fine_tuned_model.pth \
        --test_dir  path/to/test/images \
        --output_dir path/to/save/masks \
        [--device {cpu,cuda,mps}]
"""
import os
import argparse
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

def parse_args():
    parser = argparse.ArgumentParser(description="Semantic segmentation inference script")
    parser.add_argument(
        '--model_name_or_dir', required=True,
        help="Name or path of the base pretrained model (for architecture & processor)"
    )
    parser.add_argument(
        '--model_path', required=True,
        help="Path to the .pth file containing your fine‑tuned model weights"
    )
    parser.add_argument(
        '--test_dir', required=True,
        help="Directory with test images (*.jpg, *.png)"
    )
    parser.add_argument(
        '--output_dir', required=True,
        help="Where to save predicted mask images"
    )
    parser.add_argument(
        '--device', default=None, choices=['cpu', 'cuda', 'mps'],
        help="Compute device. Default auto-detects"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif getattr(torch.backends, 'mps', False) and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load processor & instantiate model architecture
    processor = AutoImageProcessor.from_pretrained(args.model_name_or_dir)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        args.model_name_or_dir,
        ignore_mismatched_sizes=True  # allows loading when num_labels differs
    )

    # Load fine‑tuned weights
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Inference over test images
    for fname in sorted(os.listdir(args.test_dir)):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        img_path = os.path.join(args.test_dir, fname)
        image = Image.open(img_path).convert('RGB')

        # Preprocess
        enc = processor(images=image, return_tensors='pt')
        pixel_values = enc['pixel_values'].to(device)

        # Forward
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            seg = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0]

        # Convert to mask and save
        mask_np = seg.cpu().numpy().astype(np.uint8)
        mask_img = Image.fromarray(mask_np, mode='L')
        out_name = os.path.splitext(fname)[0] + '_mask.png'
        mask_img.save(os.path.join(args.output_dir, out_name))
        print(f"Saved mask: {out_name}")

if __name__ == '__main__':
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
