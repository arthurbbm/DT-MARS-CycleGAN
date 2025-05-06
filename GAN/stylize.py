# #!/usr/bin/env python3
# # stylize.py
#
# import os
# import argparse
# from PIL import Image
# import torch
# from torchvision import transforms
# from transformer_net import TransformerNet
#
# def stylize(args):
#     device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
#     # 1) load model
#     transformer = TransformerNet().to(device)
#     transformer.load_state_dict(torch.load(args.model_path, map_location=device))
#     transformer.eval()
#     # 2) transforms
#     loader = transforms.Compose([
#         transforms.Resize(args.imsize),
#         transforms.CenterCrop(args.imsize),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
#     ])
#     unloader = transforms.ToPILImage()
#     os.makedirs(args.output_dir, exist_ok=True)
#     # 3) run
#     for fname in sorted(os.listdir(args.content_dir)):
#         if not fname.lower().endswith(('.jpg','.png')): continue
#         img = loader(Image.open(os.path.join(args.content_dir, fname)).convert('RGB')).unsqueeze(0).to(device)
#         with torch.no_grad():
#             out = transformer(img).clamp(0,1).cpu().squeeze(0)
#         pil = unloader(out)
#         pil.save(os.path.join(args.output_dir, fname))
#         print("Saved:", fname)
#
# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument('--content_dir', required=True)
#     p.add_argument('--model_path',  required=True)
#     p.add_argument('--output_dir',  required=True)
#     p.add_argument('--imsize',      type=int, default=512)
#     args = p.parse_args()
#     stylize(args)
#


#!/usr/bin/env python3
# stylize.py

import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from transformer_net import TransformerNet


def stylize(args):
    # select device: CUDA > MPS > CPU
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f"Using device: {device}")

    # 1) load model
    transformer = TransformerNet().to(device)
    # load weights onto the chosen device
    transformer.load_state_dict(
        torch.load(args.model_path, map_location=device)
    )
    transformer.eval()

    # 2) transforms
    loader = transforms.Compose([
        transforms.Resize(args.imsize),
        transforms.CenterCrop(args.imsize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    unloader = transforms.ToPILImage()
    os.makedirs(args.output_dir, exist_ok=True)

    # 3) run stylization on each image
    for fname in sorted(os.listdir(args.content_dir)):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        # load and preprocess
        img = loader(
            Image.open(os.path.join(args.content_dir, fname)).convert('RGB')
        ).unsqueeze(0).to(device, non_blocking=True)

        with torch.no_grad():
            out = transformer(img)
            out = out.clamp(0, 1).cpu().squeeze(0)

        # convert back to PIL and save
        pil = unloader(out)
        out_path = os.path.join(args.output_dir, fname)
        pil.save(out_path)
        print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', required=True,
                        help="directory of input images")
    parser.add_argument('--model_path', required=True,
                        help="path to transformer weights (.pth)")
    parser.add_argument('--output_dir', required=True,
                        help="where to save stylized images")
    parser.add_argument('--imsize', type=int, default=512,
                        help="image height/width")
    args = parser.parse_args()
    stylize(args)
