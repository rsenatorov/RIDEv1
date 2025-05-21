# Copyright (c) 2025, Robert Senatorov
# All rights reserved.

"""
File: src/train.py

Description:
Training script for the RIDE model using a CompositeLoss (L1 + SSIM)
with learnable loss weights, multi-scale supervision, and all of the following features:
 - Mixed precision
 - Gradient accumulation
 - Gradient clipping
 - Cosine Annealing Warm-Up (updated each batch)
 - Quick-test (~1% data)
 - Validation & test splits
 - Logging to CSV + text
 - Dynamic per-epoch loss plots
 - Resume from latest checkpoint
"""

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*spectral_angle_mapper.*"
)

import os
import re
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import csv

from data.dataset import RIDEDataset
from network.model import RIDE
from network.losses import CompositeLoss

CONFIG = {
    "train_image_dir": "dataset/images",
    "train_depth_dir": "dataset/depth",

    "batch_size": 4,
    "num_workers": 1,
    "epochs": 10,
    "lr": 1e-4,
    "weight_decay": 5e-2,
    "mixed_precision": True,
    "output_dir": "checkpoints",
    "val_split": 0.1,
    "test_split": 0.1,
    "quick_test": False,
    "prefetch_factor": 4,
    "pin_memory": False,
    "benchmark": True,
    "accumulation_steps": 2,
    "grad_clip": 1.0,

    # scheduler
    "warmup_steps": 500,
    "T_max": 5000,
}


class CosineAnnealingWarmUpScheduler(optim.lr_scheduler._LRScheduler):
    """
    Scheduler that gradually increases LR linearly during warm-up, then uses
    cosine annealing. Updates every batch step (not per epoch).
    """
    def __init__(self, optimizer, warmup_steps, T_max, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / self.T_max
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


def validate(model, data_loader, device, loss_fn, desc="Validation"):
    """
    Runs validation (or test) using CompositeLoss on final + side outputs.
    """
    model.eval()
    total_loss = 0.0
    pbar = tqdm(data_loader, desc=desc)
    with torch.inference_mode():
        for imgs, gt, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            gt   = gt.to(device, non_blocking=True)

            preds = model(imgs)
            loss = loss_fn(preds["final"], gt)
            for side in ["side4", "side3", "side2", "side1", "side0"]:
                p = preds[side]
                g = nn.functional.interpolate(gt, size=p.shape[-2:], mode="bilinear", align_corners=False)
                loss = loss + loss_fn(p, g)

            total_loss += loss.item()
            pbar.set_postfix({f"{desc.lower()}_loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(data_loader)
    model.train()
    return avg_loss


def main():
    """
    Main training function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = CONFIG["benchmark"]

    # prepare output & logs
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    training_log_file = os.path.join(logs_dir, "training_logs.txt")
    csv_file = os.path.join(logs_dir, "training_metrics.csv")

    # load full dataset
    full_ds = RIDEDataset(CONFIG["train_image_dir"], CONFIG["train_depth_dir"])
    full_size = len(full_ds)
    print(f"[INFO] Full dataset size: {full_size}")

    # quick test -> ~1%
    if CONFIG["quick_test"] and full_size > 100:
        quick_n = max(1, int(0.01 * full_size))
        full_ds = torch.utils.data.Subset(full_ds, list(range(quick_n)))
        print(f"[Quick Test] Training on {quick_n}/{full_size} samples")

    total = len(full_ds)
    test_n  = int(CONFIG["test_split"] * total)
    val_n   = int(CONFIG["val_split"]  * total)
    train_n = total - val_n - test_n
    train_ds, val_ds, test_ds = random_split(full_ds, [train_n, val_n, test_n])

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        drop_last=True,
        pin_memory=CONFIG["pin_memory"],
        prefetch_factor=CONFIG["prefetch_factor"]
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )

    # model & loss
    model   = RIDE(in_channels=3, out_size=224, pretrained=True).to(device)
    loss_fn = CompositeLoss().to(device)

    # compute parameter counts
    enc_p = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    dec_p = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    tot_p = sum(p.numel() for p in model.parameters()) / 1e6

    # optimizer: include model params + loss weight params
    optimizer = optim.AdamW(
        list(model.parameters()) + [loss_fn.s_l1, loss_fn.s_ssim],
        lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scaler    = GradScaler(enabled=CONFIG["mixed_precision"])
    
    # look for complete checkpoint files first
    complete_ckpt_files = [
        f for f in os.listdir(CONFIG["output_dir"])
        if re.match(r'ride_complete_epoch\d+\.pth$', f)
    ]
    last_epoch = 0
    scheduler_state = None
    
    if complete_ckpt_files:
        # Use the latest complete checkpoint
        last_epoch = max(
            int(re.search(r'ride_complete_epoch(\d+)\.pth', f).group(1))
            for f in complete_ckpt_files
        )
        ckpt_path = os.path.join(CONFIG["output_dir"], f"ride_complete_epoch{last_epoch}.pth")
        
        print(f"[INFO] Loading complete checkpoint from epoch {last_epoch}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Load loss function state if available
        if 'loss_fn_state_dict' in checkpoint:
            loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        
        # First set initial_lr in the optimizer before loading state dict
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = CONFIG["lr"]
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scaler state for mixed precision
        if 'scaler_state_dict' in checkpoint and CONFIG["mixed_precision"]:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Capture scheduler state to initialize it after creation
        scheduler_state = checkpoint.get('scheduler_state_dict')
        
        print(f"[INFO] Successfully loaded complete checkpoint, resuming from epoch {last_epoch+1}")
    else:
        # Fall back to model-only checkpoints
        model_ckpt_files = [
            f for f in os.listdir(CONFIG["output_dir"])
            if re.match(r'ride_epoch\d+\.pth$', f)
        ]
        if model_ckpt_files:
            last_epoch = max(
                int(re.search(r'ride_epoch(\d+)\.pth', f).group(1))
                for f in model_ckpt_files
            )
            # load only model weights
            model.load_state_dict(
                torch.load(os.path.join(CONFIG["output_dir"], f"ride_epoch{last_epoch}.pth")),
                strict=False
            )
            print(f"[INFO] Loaded model-only checkpoint (strict=False) from epoch {last_epoch}, resuming from epoch {last_epoch+1}")
            print("[WARNING] Optimizer, scheduler, and scaler states were not found and will be initialized from scratch")
        else:
            last_epoch = 0
    
    # Always initialize initial_lr for each parameter group before creating scheduler
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = CONFIG["lr"]
    
    # Create scheduler with the appropriate last_epoch value
    scheduler = CosineAnnealingWarmUpScheduler(
        optimizer, CONFIG["warmup_steps"], CONFIG["T_max"], eta_min=1e-6, last_epoch=last_epoch
    )
    
    # Load scheduler state if available
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    # write config & param counts
    if last_epoch == 0:
        with open(training_log_file, "w") as f:
            f.write("[CONFIG SETTINGS]\n")
            for k, v in CONFIG.items():
                f.write(f"{k}: {v}\n")
            f.write("\n[MODEL PARAMS]\n")
            f.write(f"Encoder: {enc_p:.2f}M\n")
            f.write(f"Decoder: {dec_p:.2f}M\n")
            f.write(f"Total:   {tot_p:.2f}M\n\n")
    else:
        with open(training_log_file, "a") as f:
            f.write(f"[INFO] Resumed training from epoch {last_epoch}\n\n")

    # prepare CSV header
    if os.path.exists(csv_file):
        with open(csv_file, "r") as f:
            metrics = list(csv.reader(f))
    else:
        metrics = [("epoch","train_loss","val_loss","test_loss","train_time","val_time","test_time")]

    # training loop
    model.train()
    start_epoch = last_epoch + 1
    end_epoch   = last_epoch + CONFIG["epochs"]

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start = time.time()
        running_loss = 0.0
        batch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch} [Train]")
        for imgs, gt, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            gt   = gt.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=CONFIG["mixed_precision"]):
                preds = model(imgs)
                loss = loss_fn(preds["final"], gt)
                for side in ["side4","side3","side2","side1","side0"]:
                    p = preds[side]
                    g = nn.functional.interpolate(gt, size=p.shape[-2:], mode="bilinear", align_corners=False)
                    loss = loss + loss_fn(p, g)
                loss = loss / CONFIG["accumulation_steps"]

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

            if (pbar.n + 1) % CONFIG["accumulation_steps"] == 0 or (pbar.n + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            scheduler.step()
            running_loss += loss.item() * CONFIG["accumulation_steps"]
            batch_losses.append(loss.item() * CONFIG["accumulation_steps"])
            pbar.set_postfix({"train_loss": f"{running_loss/(pbar.n+1):.4f}"})

        train_time = time.time() - epoch_start
        avg_train_loss = running_loss / len(train_loader)

        # save model-only checkpoint (for backward compatibility)
        ckpt_path = os.path.join(CONFIG["output_dir"], f"ride_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        
        # save complete checkpoint with all states
        complete_ckpt_path = os.path.join(CONFIG["output_dir"], f"ride_complete_epoch{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss_fn_state_dict': loss_fn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if CONFIG["mixed_precision"] else None,
            'train_loss': avg_train_loss,
        }, complete_ckpt_path)

        # validate
        val_start = time.time()
        val_loss  = validate(model, val_loader, device, loss_fn, desc="Validation")
        val_time  = time.time() - val_start

        # test only on final epoch
        if epoch == end_epoch:
            test_start = time.time()
            test_loss  = validate(model, test_loader, device, loss_fn, desc="Test")
            test_time  = time.time() - test_start
        else:
            test_loss, test_time = None, None

        # store numeric metrics (no "s" suffix!)
        metrics.append((
            epoch,
            round(avg_train_loss,4),
            round(val_loss,4),
            round(test_loss,4) if test_loss is not None else None,
            round(train_time,2),
            round(val_time,2),
            round(test_time,2) if test_time is not None else None
        ))

        # append nicely formatted epoch log
        with open(training_log_file, "a") as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"Train: {avg_train_loss:.4f} ({train_time:.2f}s)\n")
            f.write(f"Val:   {val_loss:.4f} ({val_time:.2f}s)\n")
            if test_loss is not None:
                f.write(f"Test:  {test_loss:.4f} ({test_time:.2f}s)\n")
            f.write("\n")

        # per-batch loss plot
        plt.figure()
        plt.plot(batch_losses, marker="o", linestyle="-")
        plt.title(f"Epoch {epoch} Loss per Batch")
        plt.xlabel("Batch #")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(logs_dir, f"loss_epoch_{epoch}.png"))
        plt.close()

    # write CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(metrics)

    # sum only the train_time column, casting to float to avoid mixing types
    total_time = sum(
        float(r[4]) for r in metrics[1:]
        if r[4] not in (None, "", "None")
    )
    print(f"[INFO] Training complete in {total_time:.2f}s")


if __name__ == "__main__":
    main()
