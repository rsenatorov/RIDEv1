# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/data/dataset.py

Description:
Provides the dataset class for loading 448x448 images and .npz inverse depth.
We randomly select one augmentation per base image per epoch:
 - Either full-image resize to 224x224 or one of four corner crops (TL, TR, BL, BR).
 - Randomly apply one of 4 flip modes: none, horizontal, vertical, both.
 - Apply one random color jitter, plus random y-contrast, blur & flicker augments.
Depth is returned as raw inverse-metric values (no [0,1] normalization).
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class RandomGamma:
    """
    Pickleable random gamma correction.
    """
    def __init__(self, gamma_range=(0.7, 1.3)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        return TF.adjust_gamma(img, gamma)


class RandomFlicker:
    """
    Pickleable random per-image brightness flicker.
    """
    def __init__(self, flicker_range=(0.9, 1.1)):
        self.flicker_range = flicker_range

    def __call__(self, img):
        factor = random.uniform(self.flicker_range[0], self.flicker_range[1])
        return TF.adjust_brightness(img, factor)


def resize_depth_array(depth_array, size):
    """
    Resizes a single-channel depth array to (size x size) using bilinear interpolation.
    Returns a NumPy array of floats.
    """
    depth_img = Image.fromarray(depth_array.astype(np.float32), mode="F")
    depth_img = depth_img.resize((size, size), resample=Image.BILINEAR)
    return np.array(depth_img)


class RIDEDataset(Dataset):
    """
    RIDEDataset:
      - Reads each 448x448 RGB + inverse depth
      - Produces 1 random augmentation per image per epoch:
         * Randomly choose full-image resize or one corner crop
         * Random flip
         * Random color jitter + y-contrast, blur & flicker
      - Returns raw inverse-metric depth (no [0,1] scaling).
    """
    def __init__(self, image_dir, depth_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.depth_dir = depth_dir

        # Base transform: ToTensor + ImageNet normalization
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Extended jitter: color jitter + random γ + blur + flicker
        self.jitter = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            ),
            transforms.RandomApply([RandomGamma((0.7, 1.3))], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
            transforms.RandomApply([RandomFlicker((0.9, 1.1))], p=0.5),
        ])

        # Gather valid filenames
        self.base_filenames = []
        for f in os.listdir(image_dir):
            fname, _ = os.path.splitext(f)
            npz_path = os.path.join(depth_dir, fname + ".npz")
            if os.path.isfile(npz_path):
                self.base_filenames.append(fname)
        self.base_filenames.sort()
        print(f"[RIDEDataset] Found {len(self.base_filenames)} images with matching depth.")

    def __len__(self):
        return len(self.base_filenames)

    def __getitem__(self, idx):
        fname = self.base_filenames[idx]

        # Load RGB
        img_path = os.path.join(self.image_dir, fname + ".jpg")
        if not os.path.exists(img_path):
            for ext in [".png", ".jpeg"]:
                alt = os.path.join(self.image_dir, fname + ext)
                if os.path.exists(alt):
                    img_path = alt
                    break
        with Image.open(img_path) as im_full:
            im_full = im_full.convert("RGB")

        # Load depth
        data = np.load(os.path.join(self.depth_dir, fname + ".npz"))
        depth_full = data["inverse_depth"]  # raw inverse-metric

        # Choose patch: 0 -> full resize, 1-4 -> corner crops
        crop_idx = random.randint(0, 4)
        if crop_idx == 0:
            im_patch = im_full.resize((224, 224), Image.BILINEAR)
            depth_patch = resize_depth_array(depth_full, 224)
        else:
            row = 0 if crop_idx < 3 else 224
            col = 0 if crop_idx % 2 == 1 else 224
            im_patch = im_full.crop((col, row, col + 224, row + 224))
            depth_patch = depth_full[row:row + 224, col:col + 224]

        # Random flip
        flip_idx = random.randint(0, 3)
        if flip_idx == 1:
            im_patch = im_patch.transpose(Image.FLIP_LEFT_RIGHT)
            depth_patch = np.fliplr(depth_patch).copy()
        elif flip_idx == 2:
            im_patch = im_patch.transpose(Image.FLIP_TOP_BOTTOM)
            depth_patch = np.flipud(depth_patch).copy()
        elif flip_idx == 3:
            im_patch = im_patch.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
            depth_patch = np.flipud(np.fliplr(depth_patch)).copy()

        # Jitter the RGB patch
        im_patch = self.jitter(im_patch)

        # To tensor
        img_tensor = self.transform(im_patch)
        depth_tensor = torch.from_numpy(depth_patch).unsqueeze(0).float()

        hfov_deg = float(data.get("hfov_deg", np.array([90], dtype=np.float32)))
        return img_tensor, depth_tensor, hfov_deg
