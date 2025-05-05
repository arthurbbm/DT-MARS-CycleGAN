import os
from PIL import Image
import numpy as np
import yaml
import torch
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    TrainingArguments,
    Trainer
)

# Utility to load paired image and mask files
def get_image_mask_pairs(image_dir, mask_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg'))])
    return [
        {
            'image': Image.open(os.path.join(image_dir, img)).convert('RGB'),
            'mask': Image.open(os.path.join(mask_dir, msk))
        }
        for img, msk in zip(image_files, mask_files)
    ]

# Paths
with open(os.path.join('..', 'config.yml'), 'r') as f:
    config = yaml.safe_load(f)
dataset_root = os.path.expanduser(config['segmentation_dataset'])

# Paths to your data
image_dir       = os.path.join(dataset_root, 'train', 'images')
mask_dir        = os.path.join(dataset_root, 'train', 'masks')
test_image_dir  = os.path.join(dataset_root, 'test')
test_output_dir = os.path.join(dataset_root, 'test_predictions')
os.makedirs(test_output_dir, exist_ok=True)

# Device setup
device = torch.device(
    'cuda' if torch.cuda.is_available() else
    ('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu')
)

# Data augmentations (color jitter, blur)
train_transforms = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        )
    ], p=0.8),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5)
    ], p=0.5)
])

# Load processor & model
task_model = 'google/deeplabv3_mobilenet_v2_1.0_513'
processor = AutoImageProcessor.from_pretrained(task_model)
model = AutoModelForSemanticSegmentation.from_pretrained(
    task_model,
    ignore_mismatched_sizes=True,
    num_labels=5,
    id2label={
        0: 'background',
        1: 'big_plant',
        2: 'small_plant',
        3: 'polygonum_v2',
        4: 'cirsium'
    },
    label2id={
        'background': 0,
        'big_plant': 1,
        'small_plant': 2,
        'polygonum_v2': 3,
        'cirsium': 4
    }
).to(device)

# Prepare encodings with augmentation
examples = get_image_mask_pairs(image_dir, mask_dir)
encodings = []
for ex in examples:
    # apply augmentations on the fly
    aug_img = train_transforms(ex['image'])

    # processor handles resizing + normalization
    enc = processor(images=aug_img, return_tensors='pt')
    pixel_values = enc['pixel_values'].squeeze(0).to(device)  # [C,H,W]

    # prepare mask
    _, H, W = pixel_values.shape
    mask_resized = ex['mask'].resize((W, H), resample=Image.NEAREST)
    labels = torch.tensor(np.array(mask_resized), dtype=torch.long).to(device)

    encodings.append({'pixel_values': pixel_values, 'labels': labels})

# Train/validation split
split = int(0.8 * len(encodings))
train_dataset = encodings[:split]
eval_dataset  = encodings[split:]

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    do_train=True,
    do_eval=True,
    eval_steps=100,
    save_steps=100,
    logging_dir='./logs',
    logging_steps=50,
    learning_rate=1e-4,
    save_total_limit=2,
    push_to_hub=False
)

# IoU metric
def compute_mean_iou(preds, labels, num_classes=5):
    ious = []
    for cls in range(num_classes):
        p = (preds == cls)
        t = (labels == cls)
        inter = np.logical_and(p, t).sum()
        uni = np.logical_or(p, t).sum()
        if uni > 0:
            ious.append(inter / uni)
    return float(np.mean(ious)) if ious else 0.0

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {'mean_iou': compute_mean_iou(preds, labels, num_classes=5)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics
)

# Train & save
trainer.train()
model.save_pretrained('./fine_tuned_model')
processor.save_pretrained('./fine_tuned_model')

# Inference on test images
model.eval()
with torch.no_grad():
    for fname in sorted(os.listdir(test_image_dir)):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        image = Image.open(os.path.join(test_image_dir, fname)).convert('RGB')
        enc = processor(images=image, return_tensors='pt')
        pixel_values = enc['pixel_values'].to(device)
        outputs = model(pixel_values=pixel_values)
        mask = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]]
        )[0]
        # Convert tensor to numpy array for saving
        mask_np = mask.cpu().numpy().astype(np.uint8)
        Image.fromarray(mask_np, mode='L').save(
            os.path.join(test_output_dir,
                         f"{os.path.splitext(fname)[0]}_mask.png")
        )
