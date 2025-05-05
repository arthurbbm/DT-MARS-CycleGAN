# train.py
#!/usr/bin/env python3
import os
import argparse
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

from transformer_net import TransformerNet
from losses import (
    build_style_losses,
    build_content_losses,
    build_loss_model
)

# --- Flat folder dataset (no subdirs required) ---
class FlatImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.paths = [
            os.path.join(root, fname)
            for fname in sorted(os.listdir(root))
            if fname.lower().endswith(('.jpg','jpeg','.png','.bmp','.tiff'))
        ]
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def train_transformer(args):
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    # 1) Transformer & optimizer
    transformer = TransformerNet().to(device)
    optimizer   = optim.Adam(transformer.parameters(), lr=args.lr)

    # 2) Fixed VGG for losses
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]

    # 3) Prepare style image once
    style_tf = transforms.Compose([
        transforms.Resize(args.imsize),
        transforms.CenterCrop(args.imsize),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    style = style_tf(Image.open(args.style_image).convert('RGB')).unsqueeze(0).to(device)

    # 4) Build style losses once
    style_losses = build_style_losses(
        cnn, norm_mean, norm_std,
        style, args.style_layers, device
    )

    # 5) Content dataset loader
    content_tf = transforms.Compose([
        transforms.Resize(args.imsize),
        transforms.CenterCrop(args.imsize),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    dataset = FlatImageFolder(args.content_dir, transform=content_tf)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 6) Training loop
    for epoch in range(1, args.epochs + 1):
        transformer.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(device)

            # 6a) Build content losses for this batch
            content_losses = build_content_losses(
                cnn, norm_mean, norm_std,
                batch, args.content_layers, device
            )

            # 6b) Stitch full loss model
            loss_model = build_loss_model(
                cnn, norm_mean, norm_std,
                style_losses, content_losses,
                args.style_layers, args.content_layers,
                device
            )

            optimizer.zero_grad()
            output = transformer(batch)
            output = torch.clamp(output, 0.0, 1.0)

            # 6c) Forward through loss model
            loss_model(output)

            # 6d) Sum losses
            style_score   = sum(sl.loss for _, sl in style_losses)
            content_score = sum(cl.loss for _, cl in content_losses)
            loss = style_score + args.content_weight * content_score
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs} — avg loss: {avg_loss:.4f}")

    # 7) Save trained weights
    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
    torch.save(transformer.state_dict(), args.model_path)
    print("Transformer weights saved to", args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Style Transfer Training")
    parser.add_argument('--content_dir', required=True,
                        help="folder of content images")
    parser.add_argument('--style_image', required=True,
                        help="path to style image")
    parser.add_argument('--model_path', default='transformer.pth',
                        help="where to save transformer weights")
    parser.add_argument('--imsize', type=int, default=512,
                        help="height/width for training images")
    parser.add_argument('--epochs', type=int, default=5,
                        help="number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size for training")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="learning rate for Adam optimizer")
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help="weight for content loss")
    parser.add_argument('--style_layers', nargs='+',
                        default=['conv_1','conv_2','conv_3','conv_4','conv_5'],
                        help="VGG layers for style loss")
    parser.add_argument('--content_layers', nargs='+',
                        default=['conv_4'],
                        help="VGG layers for content loss")
    args = parser.parse_args()

    train_transformer(args)
