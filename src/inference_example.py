# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/inference_example.py

Description:
Loads a trained RIDE model and visualizes the 224x224 inverse metric depth predictions
for a list of test images. Displays both the original RGB image and the predicted absolute
depth map in meters, with red indicating close and blue indicating far.
Now processes one image at a time.
"""

import os
import time
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from network.model import RIDE

# fixed HFOV for all examples
HFOV_DEG = 49.7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_IMG_SIZE = 224
CHECKPOINT_PATH = "checkpoints/ride_epoch1.pth"


def center_crop_and_resize(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Center crops and resizes an image to the target size.
    """
    h, w, _ = img.shape
    if h < target_size or w < target_size:
        return cv2.resize(img, (target_size, target_size))
    min_dim = min(h, w)
    sy = (h - min_dim) // 2
    sx = (w - min_dim) // 2
    cropped = img[sy:sy+min_dim, sx:sx+min_dim]
    return cv2.resize(cropped, (target_size, target_size))


def main():
    """
    Main function for performing inference on a set of test images using RIDE.
    """
    print(f"[INFO] Using device: {DEVICE}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Build RIDE model
    model = RIDE(in_channels=3, out_size=224, pretrained=False).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print(f"[INFO] Loaded RIDE checkpoint from {CHECKPOINT_PATH}")
    model.eval()

    test_image_files = [
        "dataset/test/1.png",
        "dataset/test/2.png",
        "dataset/test/3.png",
        "dataset/test/4.png"
    ]

    fig, axes = plt.subplots(
        nrows=len(test_image_files),
        ncols=2,
        figsize=(8, 3 * len(test_image_files))
    )
    # reversed jet: red=close, blue=far
    cmap = plt.get_cmap("jet_r")

    # precompute focal length in pixels
    hfov_rad = np.deg2rad(HFOV_DEG)
    focal_px = (TARGET_IMG_SIZE / 2) / np.tan(hfov_rad / 2)

    for idx, image_path in enumerate(test_image_files):
        print(f"[INFO] Processing {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Unable to load {image_path}")
            continue

        # Preprocess
        cropped_bgr = center_crop_and_resize(frame, TARGET_IMG_SIZE)
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        # Inference
        start_time = time.time()
        with torch.inference_mode():
            pred_inv = model(input_tensor)["final"]
        elapsed_ms = (time.time() - start_time) * 1000.0
        print(f"[INFO] Inference took {elapsed_ms:.1f} ms")

        # Raw inverse-depth prediction
        inv_map = pred_inv.squeeze(0).squeeze(0).cpu().numpy()

        # Convert to absolute depth in cm, then to meters
        depth_map = (focal_px / inv_map) / 100

        vmin, vmax = depth_map.min(), depth_map.max()
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Show RGB
        axes[idx][0].imshow(cropped_rgb)
        axes[idx][0].set_title(f"RGB: {os.path.basename(image_path)}", fontsize=9)
        axes[idx][0].axis("off")

        # Show absolute depth
        im_disp = axes[idx][1].imshow(depth_map, cmap=cmap, norm=norm)
        axes[idx][1].set_title(f"Depth (m)\n{elapsed_ms:.1f} ms", fontsize=9)
        axes[idx][1].axis("off")

        # Colorbar with metric ticks
        cbar = fig.colorbar(im_disp, ax=axes[idx][1], fraction=0.046, pad=0.04)
        ticks = np.linspace(vmin, vmax, num=5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
        cbar.set_label("Depth (m)", fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    out_path = os.path.join("logs", "inference_depth_result.png")
    plt.savefig(out_path, dpi=100)
    print(f"[INFO] Saved inference result to {out_path}")


if __name__ == "__main__":
    main()
