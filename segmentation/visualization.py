import os
from PIL import Image
import numpy as np
import yaml
import matplotlib.pyplot as plt


def get_image_mask_pairs(image_dir, mask_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg'))])
    return [
        {
            'image_path': os.path.join(image_dir, img),
            'mask_path': os.path.join(mask_dir, msk)
        }
        for img, msk in zip(image_files, mask_files)
    ]

def visualize_predictions(image_dir, mask_dir, num_to_show=5, alpha=0.5):
    pairs = get_image_mask_pairs(image_dir, mask_dir)
    colors = np.array([
        [0, 0, 0],        # background = black
        [0, 255, 0],      # big_plant = green
        [0, 0, 255],      # small_plant = blue
        [255, 0, 0],      # polygonum_v2 = red
        [255, 255, 0]     # cirsium = yellow
    ], dtype=np.uint8)

    for pair in pairs[:num_to_show]:
        img = Image.open(pair['image_path']).convert('RGB')
        mask = Image.open(pair['mask_path']).convert('L')
        mask_np = np.array(mask)
        color_mask = colors[mask_np]
        overlay = Image.blend(
            img.convert('RGBA'),
            Image.fromarray(color_mask).convert('RGBA'),
            alpha
        )
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis('off')
    plt.show()


with open("../config.yml", "r") as f:
    config = yaml.safe_load(f)
dataset_root = os.path.expanduser(config["segmentation_dataset"])

test_image_dir = os.path.join(dataset_root, 'train', 'images')
test_output_dir = os.path.join(dataset_root, 'test_predictions')
visualize_predictions(test_image_dir, test_output_dir, num_to_show=20, alpha=0.6)