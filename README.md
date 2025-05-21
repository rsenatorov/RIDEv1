# robust-image-depth-estimation (RIDEv1)

RIDEv1 is a **Robust Image Depth Estimation** framework designed to predict high-quality depth maps from a single RGB image. It leverages a ResNeXt-based encoder, a custom decoder with Atrous Spatial Pyramid Pooling (ASPP) enhanced by Strip-Pooling, and multiple refinement modules. This README covers installation, architecture, data pipeline, training, inference, and feature-visualization, and includes a **gitdiagram** section to generate a diagram of the system.

---

## 🔍 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Data Pipeline](#data-pipeline)
5. [Model Architecture](#model-architecture)

   * [Encoder (ResNeXt-101)](#encoder-resnext-101)
   * [Decoder (RIDE)](#decoder-ride)
   * [ASPP & Strip-Pooling](#aspp--strip-pooling)
   * [Auxiliary Blocks](#auxiliary-blocks)
6. [Training](#training)
7. [Inference](#inference)
8. [Feature Map Visualization](#feature-map-visualization)
9. [Gitdiagram Definition](#gitdiagram-definition)
10. [License](#license)

---

## 📝 Overview

* **Goal:** Predict per-pixel depth (as inverse-metric values) from a 224×224 RGB image.
* **Key Innovations:**

  * Multi-scale ASPP with long-strip convolutions for contextual awareness.
  * Local Planar Guidance (LPG) and auxiliary refinement for accurate fine-grained depth.
  * Hybrid skip connections with attention gating and SE blocks for effective feature fusion.
  * Composite loss combining L1 and SSIM with learnable weights.

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ride.git
cd ride

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

Dependencies include:

* PyTorch (≥1.12)
* torchvision
* torchmetrics
* OpenCV
* matplotlib
* tqdm

---

## 📁 Project Structure

```
ride/
├── data/
│   └── dataset.py           # RIDEDataset: augmentations & loading
├── network/
│   ├── encoder.py           # ResNeXt-101 encoder
│   ├── decoder.py           # RIDE decoder orchestration
│   ├── model.py             # RIDE wrapper
│   └── blocks/              # Building blocks (ASPP, LPG, SE, etc.)
├── checkpoints/             # Saved .pth checkpoints
├── logs/
│   ├── training_metrics.csv # Training logs
│   └── features/            # ASPP feature-map PNGs
├── src/
│   ├── train.py             # Training script
│   ├── inference_example.py # Batch-offline inference
│   └── inference_webcam.py  # Real-time webcam demo
└── extract_aspp_feature_maps.py # Utility: visualize ASPP channels
```

---

## 🔄 Data Pipeline

1. **Image & Depth Loading** (`RIDEDataset`):

   * Reads 448×448 RGB + .npz inverse-depth maps.
   * Random crop or resize to 224×224.
   * Random flips, color jitter, gamma, blur, flicker.
   * Returns: `(img_tensor, depth_tensor, hfov_deg)`.
2. **Normalization:** ImageNet mean/std; depth left unnormalized (raw meters⁻¹).

---

## 🏗️ Model Architecture

### Encoder (ResNeXt-101)

* Pretrained on ImageNet.
* Outputs multi-scale features at 1/2, 1/4, 1/8, 1/16, 1/32 resolutions.

### Decoder (RIDE)

* **UpsampleConv**: Bilinear upsample + 3×3 conv + ReLU.
* **HybridSkipBlock**: Attention-gated skip + SE channel recalibration.
* **EdgeAttentionBlock**: Gated refinement using encoder edges.
* **ASPPModule**: Atrous rates (1,6,12,18) + global pooling + 1×K/K×1 strips.
* **ResidualBlocks**: Two res blocks on ASPP output.
* **Coarse Depth Head**: 3×3 conv + learnable scale/shift.
* **LPG (Local Planar Guidance)**: Plane parameterization fused per-pixel.
* **AuxRefinementBlock**: Predict residual + gate to correct fine details.
* **Side Outputs**: Multi-scale supervision at intermediate scales.

### Loss: CompositeLoss

* L1 + (1−SSIM), each weighted by learnable log-parameters.

---

## 🚀 Training

```bash
python src/train.py --config configs/train.yaml
```

* Mixed-precision, gradient accumulation, cosine warmup scheduler.
* Splits: train/val/test (default 80/10/10).
* Checkpoints: `ride_epoch{n}.pth` & `ride_complete_epoch{n}.pth`.
* Logging: CSV & per-epoch loss curves.

---

## 🎯 Inference

* **Offline Batch**: `src/inference_example.py`
* **Real-Time Webcam**: `src/inference_webcam.py`

Both scripts:

* Center-crop + resize.
* Normalize & infer.
* Convert inverse-depth → meters.
* Visualize with `jet_r` colormap and colorbars.

---

## 🔬 Feature Map Visualization

Utility: `extract_aspp_feature_maps.py`.

* Hooks ASPP 1×1 branch, grabs post-BN activations.
* Processes 4 test images, saves 256 PNGs (4×2 grid) under `logs/features/`.
* Progress bar via `tqdm`.

---

## 📈 Gitdiagram Definition

Use the following block in your README to generate an architecture diagram with **gitdiagram**:

```gitdiagram
# Nodes
Dataset:RIDEDataset --|> Encoder:ResNeXt101
Encoder --> Decoder:RIDE
Decoder --> Blocks:ASPPModule
Decoder --> Blocks:HybridSkipBlock
Decoder --> Blocks:EdgeAttentionBlock
Decoder --> Blocks:UpsampleConv
Decoder --> Blocks:LPG
Decoder --> Blocks:AuxRefinementBlock
Decoder --> Heads:CoarseDepth + SideOutputs

# Flows
Client -> Dataset -> Model[Encoder+Decoder] -> Prediction
InferenceExample -> Client
Webcam -> Client
extract_aspp_feature_maps -> Model
```

This describes modules and their relationships; **gitdiagram** will translate it into a visual flow.

---

## 📜 License

Review the MIT license
