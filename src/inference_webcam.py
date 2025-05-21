# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/inference_webcam.py

Description:
Real-time webcam demo that runs the RIDE model on each frame
to produce 224x224 absolute depth in meters. Displays both the RGB
input and the generated depth map side by side, with red indicating
close and blue indicating far, plus a 5-tick colorbar (meters) on the right.
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
from network.model import RIDE

# fixed HFOV for all frames
HFOV_DEG = 70.4 # put your cameras hfov here in degrees

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_INPUT_SIZE = 224
DISPLAY_SIZE = 224
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
    Real-time webcam demo with absolute metric depth and colorbar.
    """
    print(f"[INFO] Using device: {DEVICE}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load model
    model = RIDE(in_channels=3, out_size=MODEL_INPUT_SIZE, pretrained=False).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print(f"[INFO] Loaded RIDE checkpoint from {CHECKPOINT_PATH}")
    model.eval()

    # compute focal length in pixels from HFOV
    hfov_rad = np.deg2rad(HFOV_DEG)
    focal_px = (MODEL_INPUT_SIZE / 2) / np.tan(hfov_rad / 2)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam.")
    window = "RIDE Webcam: RGB + Depth (m)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 800, 400)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # prepare input
        cropped = center_crop_and_resize(frame, MODEL_INPUT_SIZE)
        rgb_model = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_model)
        tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        # Inference
        start = time.time()
        with torch.inference_mode():
            pred_inv = model(tensor)["final"]
        elapsed = time.time() - start
        fps = 1.0 / elapsed if elapsed > 0 else float("inf")

        inv_map = pred_inv.squeeze(0).squeeze(0).cpu().numpy()
        # DepthPro style + mm→m: Z_mm = focal_px / inv_map; Z_m = Z_mm / 100
        depth_map = (focal_px / inv_map) / 100

        # Normalize for display on reversed jet
        vmin, vmax = depth_map.min(), depth_map.max()
        normed = np.clip((depth_map - vmin) / (vmax - vmin), 0.0, 1.0)
        depth_uint8 = np.uint8(255 * normed)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)  # BGR
        depth_color = depth_color[:, :, ::-1]  # swap BGR→RGB to get red=close, blue=far

        # resize for consistent display
        depth_vis = cv2.resize(depth_color, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
        rgb_disp = cv2.resize(cropped, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_LINEAR)
        cv2.putText(rgb_disp, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # build colorbar: 5‐tick gradient
        bar_height = DISPLAY_SIZE
        bar_color_width = 20
        bar_label_width = 50
        bar_total_width = bar_color_width + bar_label_width

        # create gradient from far (255) at top to close (0) at bottom
        gradient = np.linspace(255, 0, bar_height, dtype=np.uint8)
        # apply colormap & swap channels to match depth_vis
        cm = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)[:, 0, :]  # (H,3) BGR
        cm = cm[:, ::-1]  # swap BGR→RGB
        # build full bar
        bar_vis = np.zeros((bar_height, bar_total_width, 3), dtype=np.uint8)
        bar_vis[:, :bar_color_width, :] = np.repeat(cm[:, np.newaxis, :], bar_color_width, axis=1)

        # compute tick values & positions
        ticks = np.linspace(vmin, vmax, 5)
        for i, val in enumerate(ticks):
            # y position from bottom=0 to top=4
            y = int((bar_height - 1) - i * (bar_height - 1) / 4)
            # draw tick mark
            cv2.line(bar_vis, (bar_color_width - 1, y), (bar_color_width + 4, y), (255, 255, 255), 1)
            # label to the right
            label = f"{val:.2f} m"
            cv2.putText(bar_vis, label, (bar_color_width + 6, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # combine all panels
        combined = cv2.hconcat([rgb_disp, depth_vis, bar_vis])
        cv2.imshow(window, combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed.")

if __name__ == "__main__":
    main()
