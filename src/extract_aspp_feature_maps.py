#!/usr/bin/env python3
"""
File: extract_aspp_feature_maps.py

Loads a trained RIDE model checkpoint (ride_epoch1.pth), extracts the ASPP 1x1 filters
(after BatchNorm but before ReLU), and for each of the 256 channels, generates a matplotlib
figure showing the 4 input RGB images on the left and their corresponding heatmap activations
on the right (4 rows × 2 columns). Saves one PNG per channel into logs/features/, mirroring
the organization style of inference_example.py, with a tqdm progress bar.
"""
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from network.model import RIDE

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_IMG_SIZE = 224
CHECKPOINT_PATH = "checkpoints/ride_epoch1.pth"
TEST_IMAGE_DIR = "dataset/test"
OUTPUT_DIR = "logs/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Center-crop & resize utility
def center_crop_and_resize(img: np.ndarray, target_size: int = TARGET_IMG_SIZE) -> np.ndarray:
    h, w, _ = img.shape
    if h < target_size or w < target_size:
        return cv2.resize(img, (target_size, target_size))
    m = min(h, w)
    sy = (h - m) // 2
    sx = (w - m) // 2
    cropped = img[sy:sy+m, sx:sx+m]
    return cv2.resize(cropped, (target_size, target_size))

# Main extraction
def main():
    # Prepare test images
    test_paths = sorted(
        os.path.join(TEST_IMAGE_DIR, f)
        for f in os.listdir(TEST_IMAGE_DIR)
        if f.lower().endswith(('.png','.jpg','.jpeg'))
    )[:4]
    orig_rgbs = []
    proc_tensors = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    for path in test_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        cropped = center_crop_and_resize(img)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        orig_rgbs.append(rgb)
        proc_tensors.append(transform(Image.fromarray(rgb)))
    batch = torch.stack(proc_tensors, dim=0).to(DEVICE)

    # Load model
    model = RIDE(in_channels=3, out_size=TARGET_IMG_SIZE, pretrained=False).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Hook ASPP 1x1 conv BN output
    features = []
    def hook_fn(module, inp, out):
        # out shape: [B,256,H,W]
        features.append(F.relu(out).detach().cpu().numpy())
    handle = model.decoder.aspp.bn1.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(batch)
    handle.remove()

    fmap = features[0]  # numpy array shape (4,256,224,224)
    cmap = plt.get_cmap("jet_r")

    # Iterate channels
    for ch in tqdm(range(fmap.shape[1]), desc="Saving features"):
        # Setup figure 4 rows × 2 cols
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(6,12))
        # Gather channel activations for normalization
        ch_maps = fmap[:, ch, :, :]
        vmin, vmax = ch_maps.min(), ch_maps.max()
        norm = Normalize(vmin=vmin, vmax=vmax)

        for i in range(4):
            # Left: original RGB
            axes[i, 0].imshow(orig_rgbs[i])
            axes[i, 0].set_title(f"RGB: {os.path.basename(test_paths[i])}")
            axes[i, 0].axis('off')

            # Right: heatmap activation
            axes[i, 1].imshow(ch_maps[i], cmap=cmap, norm=norm)
            axes[i, 1].set_title(f"Feature {ch:03d}")
            axes[i, 1].axis('off')
            # Add colorbar on right of each row
            cbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=axes[i,1], fraction=0.046, pad=0.04
            )
            cbar.ax.tick_params(labelsize=6)

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"feature_map_{ch:03d}.png")
        plt.savefig(out_path, dpi=100)
        plt.close(fig)

    print(f"[INFO] Saved {fmap.shape[1]} feature visualizations to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
