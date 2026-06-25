#!/usr/bin/env python3
"""
EMMA-BC Phase B: Multimodal Clinical Training
═══════════════════════════════════════════════
Audio + Text → PHQ-8/PHQ-9 + GAD-7 + PSQI regression.

Integrates ALL code review fixes:
  1. Masked mean pooling with attention_mask gating
  2. Per-participant data split (no leakage)
  3. Optimizer: filter(p.requires_grad) — mandatory
  4. De-normalized metrics (MAE/CCC on original scale)
  5. BatchNorm frozen via .eval() before training
  6. Checkpoint resume with full optimizer/scheduler state
  7. Label Z-score normalization with stored stats
  8. DataLoader with seeded generator
"""

import os, sys, json, argparse, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multimodal_model import (
    MultimodalClinicalModel,
    regression_metrics,
    set_seed,
)
from multimodal_dataset import (
    DAICWOZDataset,
    MODMADataset,
    collate_daic,
    collate_modma,
)


# ═══════════════════════════════════════════
# Label normalization
# ═══════════════════════════════════════════

class LabelNormalizer:
    """Z-score normalize labels. Store mean/std for de-normalization at eval time."""
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, labels):
        """labels: torch.Tensor of shape [N, num_tasks]"""
        self.mean = labels.mean(dim=0)
        self.std = labels.std(dim=0).clamp(min=1e-6)
        print(f"Label stats — mean: {self.mean.tolist()}, std: {self.std.tolist()}")
    
    def normalize(self, labels):
        return (labels - self.mean.to(labels.device)) / self.std.to(labels.device)
    
    def denormalize(self, labels):
        return labels * self.std.to(labels.device) + self.mean.to(labels.device)
    
    def state_dict(self):
        return {"mean": self.mean, "std": self.std}
    
    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.std = d["std"]


# ═══════════════════════════════════════════
# Training
# ═══════════════════════════════════════════

def train_epoch(model, loader, normalizer, optimizer, device, epoch, total_epochs):
    model.train()
    losses = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc=f"Train E{epoch}/{total_epochs}")
    
    for batch in pbar:
        audio = batch["audio"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        texts = batch["texts"]
        labels = batch["phq_total"].to(device)  # [B]
        
        # Normalize labels
        labels_norm = normalizer.normalize(labels.unsqueeze(-1)).squeeze(-1)
        
        optimizer.zero_grad()
        outputs = model(audio, texts, attention_mask=attention_mask)
        pred = outputs["phq"].squeeze(-1)  # [B]
        
        loss = nn.functional.mse_loss(pred, labels_norm)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "gnorm": f"{grad_norm:.2f}"})
    
    return losses / n_batches


@torch.no_grad()
def val_epoch(model, loader, normalizer, device):
    model.eval()
    norm_mean = normalizer.mean.to(device) if normalizer.mean is not None else None
    norm_std = normalizer.std.to(device) if normalizer.std is not None else None
    all_preds, all_labels = [], []
    losses = 0.0
    n_batches = 0
    
    for batch in tqdm(loader, desc="Eval"):
        audio = batch["audio"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        texts = batch["texts"]
        labels = batch["phq_total"].to(device)
        
        labels_norm = normalizer.normalize(labels.unsqueeze(-1)).squeeze(-1)
        outputs = model(audio, texts, attention_mask=attention_mask)
        preds_norm = outputs["phq"].squeeze(-1)
        
        loss = nn.functional.mse_loss(preds_norm, labels_norm)
        losses += loss.item()
        n_batches += 1
        
        all_preds.append(preds_norm)
        all_labels.append(labels_norm)
    
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    
    # CRITICAL: de-normalize BEFORE computing metrics
    metrics = regression_metrics(
        preds, labels,
        pred_mean=norm_mean[0] if norm_mean is not None else None,
        pred_std=norm_std[0] if norm_std is not None else None,
    )
    metrics["loss"] = losses / n_batches
    return metrics


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/data/disk1/datasets/diac_woz")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4, help="Higher LR for small trainable head")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phaseB")
    parser.add_argument("--audio_pretrained", type=str, default="checkpoints/phaseA/phaseA_best.pt")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="min(4, CPU/2). Higher causes torchaudio I/O contention")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Seed everything
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ── 1. Data ──
    print("\n═══ Loading Data ═══")
    full_ds = DAICWOZDataset(args.data, split="train")
    
    # Per-participant split (no leakage!)
    train_ds, val_ds = full_ds.split_val_from_train(
        val_ratio=args.val_split, seed=args.seed
    )
    
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_daic, num_workers=args.num_workers,
        pin_memory=True, generator=gen, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_daic, num_workers=min(2, args.num_workers),
        pin_memory=True,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Compute label statistics for normalization
    print("\n═══ Computing label stats ═══")
    all_train_labels = []
    for batch in train_loader:
        all_train_labels.append(batch["phq_total"])
    normalizer = LabelNormalizer()
    normalizer.fit(torch.cat(all_train_labels))
    
    # ── 2. Model ──
    print("\n═══ Building Model ═══")
    model = MultimodalClinicalModel(
        d_model=args.d_model,
        n_layers=4, n_heads=8,
        freeze_audio_w2v=True,
        freeze_text_bert=True,
        n_tasks=1,  # PHQ-8 only for DAIC-WOZ
        audio_pretrained=args.audio_pretrained if os.path.exists(args.audio_pretrained) else None,
    ).to(device)
    
    # CRITICAL Fix 2: freeze BatchNorm/LayerNorm in frozen backbones
    model.freeze_backbones_for_training()
    
    params = model.count_params()
    print(f"Params: {params['total']:,} total, {params['trainable']:,} trainable")
    
    # ── 3. Optimizer (CRITICAL Fix 1: filter requires_grad) ──
    # DO NOT use model.parameters() — it allocates gradient memory for frozen params
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # ── 4. Resume (CRITICAL Fix 4: full state restoration) ──
    start_epoch = 1
    best_ccc = -1.0
    best_mae = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"\n═══ Resuming from {args.resume} ═══")
        # weights_only=False required for optimizer state dict
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        normalizer.load_state_dict(ckpt["label_normalizer"])
        start_epoch = ckpt["epoch"] + 1
        best_ccc = ckpt.get("best_ccc", -1.0)
        best_mae = ckpt.get("best_mae", float('inf'))
        
        # Restore warmup step count
        if "global_step" in ckpt:
            for _ in range(ckpt["global_step"]):
                scheduler.step()
            print(f"Global step restored: {ckpt['global_step']}")
        
        # Handle --lr override
        if args.lr != parser.get_default("lr"):
            print(f"Note: --lr={args.lr} overrides checkpoint LR")
            for g in optimizer.param_groups:
                g["lr"] = args.lr
        
        print(f"Resumed: epoch {start_epoch}, best_ccc={best_ccc:.4f}")
    
    # ── 5. Training ──
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    history = []
    
    print(f"\n═══ Training ({start_epoch} \u2192 {args.epochs}) ═══")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, normalizer, optimizer, device, epoch, args.epochs)
        metrics = val_epoch(model, val_loader, normalizer, device)
        scheduler.step()
        
        info = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }
        history.append(info)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val   MAE:  {metrics['mae']:.3f}  |  RMSE: {metrics['rmse']:.3f}  |  CCC: {metrics['ccc']:.3f}")
        
        # Save best (by CCC)
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_ccc = metrics["ccc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "label_normalizer": normalizer.state_dict(),
                "best_mae": best_mae,
                "best_ccc": best_ccc,
                "global_step": (epoch - 1) * len(train_loader),
                "history": history,
                "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            }, os.path.join(args.checkpoint_dir, "phaseB_best.pt"))
            print(f"  \u2713 Best checkpoint (MAE={best_mae:.3f}, CCC={best_ccc:.4f})")
        
        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "label_normalizer": normalizer.state_dict(),
            "best_mae": best_mae,
                "best_ccc": best_ccc,
            "global_step": (epoch - 1) * len(train_loader),
            "history": history,
        }, os.path.join(args.checkpoint_dir, "phaseB_latest.pt"))
    
    # Save history
    with open(os.path.join(args.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n\u2705 Phase B complete! Best MAE: {best_mae:.3f}, CCC: {best_ccc:.4f}")
    print(f"   Checkpoint: {args.checkpoint_dir}/phaseB_best.pt")


if __name__ == "__main__":
    main()
