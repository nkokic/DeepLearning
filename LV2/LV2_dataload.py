## Importing useful libraries
import os
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image

import matplotlib.pyplot as plt
from imageio.v2 import imread

from tqdm.notebook import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
print("Device", device)

import torchvision.transforms as transforms

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class IrisPupilEyeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, num_classes=4, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        
        # Get list of image files (assuming they have the same filename in both directories)
        self.image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png') or fname.endswith('.jpg')])
        self.mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('.png') or fname.endswith('.jpg')])
        
        # Ensure the number of images matches the number of masks
        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks must be the same."
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert('RGB')  # Ensure the image is in RGB
        mask = Image.open(self.mask_paths[idx]).convert('L')  # Convert mask to grayscale ('L')

        # Convert mask to class indices (this is the required format for CrossEntropyLoss)
        mask = np.array(mask)  # Shape: (height, width)

        # If one-hot encoding is required for the model output, you can still do it later
        mask = torch.tensor(mask, dtype=torch.long)  # Convert mask to tensor of integers (class indices)

        if self.transform:
            # Apply transformations to both image and mask
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Define transformations (both for images and masks)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize both images and masks to 128x128
    transforms.ToTensor(),  # Convert both images and masks to tensor format
])

# Paths to the image and mask directories
train_images_dir = 'iris_pupil_eye/train/images/'
train_masks_dir = 'iris_pupil_eye/train/masks/'

val_images_dir = 'iris_pupil_eye/val/images/'
val_masks_dir = 'iris_pupil_eye/val/masks/'

test_images_dir = 'iris_pupil_eye/test/images/'
test_masks_dir = 'iris_pupil_eye/test/masks/'

# Create dataset instances
train_dataset = IrisPupilEyeDataset(image_dir=train_images_dir, mask_dir=train_masks_dir, transform=transform)
val_dataset = IrisPupilEyeDataset(image_dir=val_images_dir, mask_dir=val_masks_dir, transform=transform)
test_dataset = IrisPupilEyeDataset(image_dir=test_images_dir, mask_dir=test_masks_dir, transform=transform)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("DataLoaders created.")
print(f"Train loader has {len(train_loader)} batches.")
print(f"Test loader has {len(test_loader)} batches.")
print(f"Validation loader has {len(val_loader)} batches.")

# Example: Show 3 random images with their masks
def show_random_images(dataset, num_images=3):
    random_indices = random.sample(range(len(dataset)), num_images)
    images_to_show = [dataset[i][0] for i in random_indices]
    masks_to_show = [dataset[i][1] for i in random_indices]

    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5))
    for i in range(num_images):
        # Convert to numpy for matplotlib display
        ax_img, ax_mask = axes[i]
        
        ax_img.imshow(images_to_show[i].permute(1, 2, 0))  # Image tensor: CxHxW -> HxWxC
        ax_img.set_title(f"Image {i+1}")
        ax_img.axis('off')
        
        ax_mask.imshow(masks_to_show[i].squeeze(0), cmap='gray')  # Mask tensor: CxHxW -> HxW
        ax_mask.set_title(f"Mask {i+1}")
        ax_mask.axis('off')

    plt.show()

# Show random images and their corresponding masks
# show_random_images(train_dataset, num_images=3)