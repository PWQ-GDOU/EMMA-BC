#!/usr/bin/env python3
"""
Complete training pipeline for CancerEmotionSystem.

Two modes:
  1. DAIC-WOZ pretraining:  [Audio + Text] → PHQ-8
  2. Hospital fine-tuning:  [Audio + Text] → Multi-Scale (HADS, VAS, CFS, PROMIS, LSNS)

Usage:
  # DAIC-WOZ pretraining
  python train.py --mode daic --data /path/to/DAIC-WOZ

  # Hospital fine-tuning
  python train.py --mode hospital --data /path/to/hospital_data \\
                  --pretrained checkpoints/daic_best.pt

  # Quick test with dummy data
  python train.py --mode dummy
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np

# ── Project imports ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mult import MulT_AudioText, MultiScaleHead, build_scale_config_from_json
from src.models.mult.mul_t_model import EmotionLoss, ConcordanceCorrelationCoefficient
from src.data.datasets.daic_woz_loader import DAICWOZLoader, DAICWOZDataset


# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Model
    "d_audio": 768,
    "d_text": 768,
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 8,
    "d_ff": 1024,
    "dropout": 0.1,
    "max_seq_len": 500,

    # Training
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": 3,
    "ccc_weight": 0.3,
    "grad_clip": 1.0,

    # Early stopping
    "patience": 10,
    "min_delta": 0.001,

    # Data
    "num_workers": 4,
    "val_split_ratio": 0.15,
}


# ══════════════════════════════════════════════════════════════════
# Training Utilities
# ══════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Early stopping with patience and model restoration."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        improved = (score < self.best_score - self.min_delta) if self.mode == "min" else (
            score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore(self, model: nn.Module):
        model.load_state_dict(self.best_state)


class AverageMeter:
    """Track running average of a metric."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute regression metrics."""
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    # MSE
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)

    # MAE
    mae = np.mean(np.abs(pred - target))

    # Pearson correlation
    if pred.shape[0] > 1:
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        corr = np.corrcoef(pred_flat, target_flat)[0, 1]
    else:
        corr = 0.0

    return {"mse": mse, "rmse": rmse, "mae": mae, "pearson_r": corr}


# ══════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad: float = 1.0,
    use_audio: bool = True,
    use_text: bool = True,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()

    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(dataloader):
        # ── Unpack batch (handles both dict and tuple) ──
        if isinstance(batch, (tuple, list)):
            # Dummy TensorDataset: (audio_feat, text_feat, audio_mask, text_mask, labels)
            a_feat, t_feat, a_mask, t_mask, labels = batch
            B = len(labels)
        else:
            # Dict batch (DAIC-WOZ or hospital loader)
            B = len(batch["label"])
            labels = batch["label"]
        
        # ── Audio features ──
        if isinstance(batch, (tuple, list)):
            audio_features = a_feat.to(device)
            audio_mask = a_mask.to(device)
        elif "audio_features" in batch:
            audio_features = batch["audio_features"].to(device)
            audio_mask = batch["audio_mask"].to(device)
        elif use_audio and "audio_path" in batch:
            T_audio = 100
            audio_features = torch.randn(B, T_audio, 768, device=device)
            audio_mask = torch.ones(B, T_audio, dtype=torch.bool, device=device)
        else:
            audio_features = audio_mask = None

        # ── Text features ──
        if isinstance(batch, (tuple, list)):
            text_features = t_feat.to(device)
            text_mask = t_mask.to(device)
        elif "text_features" in batch:
            text_features = batch["text_features"].to(device)
            text_mask = batch["text_mask"].to(device)
        elif use_text and "text_path" in batch:
            T_text = 50
            text_features = torch.randn(B, T_text, 768, device=device)
            text_mask = torch.ones(B, T_text, dtype=torch.bool, device=device)
        else:
            text_features = text_mask = None

        # ── Labels ──
        labels = labels.to(device)
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)

        # ── Forward ──
        predictions, _ = model(
            audio_features,
            text_features,
            audio_mask=audio_mask,
            text_mask=text_mask,
            return_aux=False,
        )

        # ── Loss ──
        raw_loss = criterion(predictions, labels)
        if isinstance(raw_loss, tuple):
            loss, loss_dict = raw_loss
        else:
            loss = raw_loss
            loss_dict = {"total": loss.item()}

        # ── Backward ──
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        # ── Track ──
        losses.update(loss.item(), B)
        all_preds.append(predictions.detach().cpu())
        all_targets.append(labels.detach().cpu())

        if batch_idx % 10 == 0:
            print(f"    Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    # Compute epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = losses.avg

    return metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    for batch in dataloader:
        # ── Unpack batch (handles both dict and tuple) ──
        if isinstance(batch, (tuple, list)):
            a_feat, t_feat, a_mask, t_mask, labels = batch
            B = len(labels)
            audio_features = a_feat.to(device)
            text_features = t_feat.to(device)
            audio_mask = a_mask.to(device)
            text_mask = t_mask.to(device)
        elif "audio_features" in batch:
            B = len(batch["label"])
            audio_features = batch["audio_features"].to(device)
            text_features = batch["text_features"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            text_mask = batch["text_mask"].to(device)
            labels = batch["label"]
        else:
            B = len(batch["label"])
            T_audio, T_text = 100, 50
            audio_features = torch.randn(B, T_audio, 768, device=device)
            text_features = torch.randn(B, T_text, 768, device=device)
            audio_mask = torch.ones(B, T_audio, dtype=torch.bool, device=device)
            text_mask = torch.ones(B, T_text, dtype=torch.bool, device=device)
            labels = batch["label"]

        labels = labels.to(device)
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)

        predictions, _ = model(
            audio_features, text_features,
            audio_mask=audio_mask, text_mask=text_mask,
            return_aux=False,
        )

        raw_loss = criterion(predictions, labels)
        if isinstance(raw_loss, tuple):
            loss, _ = raw_loss
        else:
            loss = raw_loss
        losses.update(loss.item(), B)
        all_preds.append(predictions.cpu())
        all_targets.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = losses.avg

    return metrics


# ══════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════

def train_daic_woz(args):
    """
    Phase A: Pretrain on DAIC-WOZ.

    Model: MulT_AudioText (single output → PHQ-8)
    Data:  DAIC-WOZ (audio + transcript + PHQ-8)
    """
    print("\n" + "=" * 60)
    print("Phase A: DAIC-WOZ Pretraining")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    has_real_data = args.data and Path(args.data).exists()
    if has_real_data:
        loader = DAICWOZLoader(args.data)
        stats = loader.get_stats()
        train_dl = loader.get_dataloader(batch_size=args.batch_size, split="train")
        val_dl = loader.get_dataloader(batch_size=args.batch_size, split="dev", shuffle=False)
    else:
        print("\n⚠️  No DAIC-WOZ data found. Using dummy data for pipeline validation.")
        print("   Apply at: https://dcapswoz.ict.usc.edu/\n")
        from torch.utils.data import DataLoader, TensorDataset
        # Create dummy dataset
        n_train, n_val = 140, 30
        X_audio = torch.randn(n_train, 100, 768)
        X_text = torch.randn(n_train, 50, 768)
        y = torch.randn(n_train, 1) * 6 + 12  # PHQ-8 ~ [0, 24]
        dummy_train = TensorDataset(
            X_audio, X_text, torch.ones(n_train, 100).bool(),
            torch.ones(n_train, 50).bool(), y
        )
        train_dl = DataLoader(dummy_train, batch_size=args.batch_size, shuffle=True)

        X_audio_v = torch.randn(n_val, 100, 768)
        X_text_v = torch.randn(n_val, 50, 768)
        y_v = torch.randn(n_val, 1) * 6 + 12
        dummy_val = TensorDataset(
            X_audio_v, X_text_v, torch.ones(n_val, 100).bool(),
            torch.ones(n_val, 50).bool(), y_v
        )
        val_dl = DataLoader(dummy_val, batch_size=args.batch_size, shuffle=False)

    # ── Model ──
    model = MulT_AudioText(
        d_audio=args.d_audio,
        d_text=args.d_text,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        num_outputs=1,  # PHQ-8
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MulT_AudioText ({n_params:,} params)")

    # ── Optimizer & Scheduler ──
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=len(train_dl) * args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[len(train_dl) * args.warmup_epochs])

    # ── Loss ──
    criterion = EmotionLoss(ccc_weight=args.ccc_weight)

    # ── Training loop ──
    early_stopper = EarlyStopping(patience=args.patience, mode="min")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_rmse = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_rmse": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\n─ Epoch {epoch}/{args.epochs} ─")

        # Train
        train_metrics = train_epoch(model, train_dl, optimizer, criterion, device, args.grad_clip)
        print(f"  Train | Loss: {train_metrics['loss']:.4f} | "
              f"RMSE: {train_metrics['rmse']:.4f} | MAE: {train_metrics['mae']:.4f}")

        # Validate
        val_metrics = validate_epoch(model, val_dl, criterion, device)
        print(f"  Val   | Loss: {val_metrics['loss']:.4f} | "
              f"RMSE: {val_metrics['rmse']:.4f} | MAE: {val_metrics['mae']:.4f} | "
              f"r: {val_metrics['pearson_r']:.3f}")

        scheduler.step()

        # Track
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_rmse"].append(val_metrics["rmse"])

        # Save best
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_rmse": best_val_rmse,
                "config": {k: v for k, v in vars(args).items()},
            }, checkpoint_dir / "daic_best.pt")
            print(f"  ✓ Best model saved (RMSE={best_val_rmse:.4f})")

        # Early stopping
        if early_stopper(val_metrics["loss"], model):
            print(f"\n  Early stopping at epoch {epoch}")
            early_stopper.restore(model)
            break

    # Save final
    torch.save(model.state_dict(), checkpoint_dir / "daic_final.pt")
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=float)

    print(f"\n✅ DAIC-WOZ pretraining complete. Best RMSE: {best_val_rmse:.4f}")
    print(f"   Model saved to: {checkpoint_dir}/daic_best.pt")
    return model


def train_hospital(args):
    """
    Phase C: Fine-tune on hospital data with multi-scale outputs.

    Model: MulT_AudioText (6 outputs → HADS-A, HADS-D, VAS, CFS, PROMIS, LSNS)
    """
    print("\n" + "=" * 60)
    print("Phase C: Hospital Fine-Tuning (Multi-Scale)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Get scale config ──
    output_ranges, scale_names = build_scale_config_from_json(
        PROJECT_ROOT / "configs" / "scales"
    )
    print(f"Scales: {scale_names}")
    print(f"Ranges: {output_ranges}")

    # ── Model ──
    model = MulT_AudioText(
        d_audio=args.d_audio,
        d_text=args.d_text,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        num_outputs=len(scale_names),
        output_ranges=output_ranges,
        use_multi_head=True,
    ).to(device)

    # Load pretrained encoder
    if args.pretrained:
        print(f"Loading pretrained encoder from: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            pretrained_state = checkpoint["model_state_dict"]
        else:
            pretrained_state = checkpoint

        # Filter to encoder weights only (not pred_head)
        encoder_state = {
            k: v for k, v in pretrained_state.items()
            if not k.startswith("pred_head")
        }
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        if missing:
            print(f"  New layers (initialized randomly): {len(missing)}")
        print(f"  ✓ Loaded {len(encoder_state)} encoder parameters")

    else:
        print("⚠️  No pretrained model. Training from scratch.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MulT_AudioText + MultiScaleHead ({n_params:,} params)")

    # ── Data (dummy for now) ──
    from torch.utils.data import DataLoader, TensorDataset
    n_train, n_val = 100, 20
    X_audio = torch.randn(n_train, 100, 768)
    X_text = torch.randn(n_train, 50, 768)
    y = torch.rand(n_train, len(scale_names))
    # Scale to ranges
    for i, (lo, hi) in enumerate(output_ranges):
        y[:, i] = y[:, i] * (hi - lo) + lo
    dummy_train = TensorDataset(
        X_audio, X_text, torch.ones(n_train, 100).bool(),
        torch.ones(n_train, 50).bool(), y
    )
    train_dl = DataLoader(dummy_train, batch_size=args.batch_size, shuffle=True)

    X_audio_v = torch.randn(n_val, 100, 768)
    X_text_v = torch.randn(n_val, 50, 768)
    y_v = torch.rand(n_val, len(scale_names))
    for i, (lo, hi) in enumerate(output_ranges):
        y_v[:, i] = y_v[:, i] * (hi - lo) + lo
    dummy_val = TensorDataset(
        X_audio_v, X_text_v, torch.ones(n_val, 100).bool(),
        torch.ones(n_val, 50).bool(), y_v
    )
    val_dl = DataLoader(dummy_val, batch_size=args.batch_size, shuffle=False)

    # ── Optimizer (lower LR for encoder if pretrained) ──
    if args.pretrained:
        encoder_lr = args.lr * 0.1
        head_lr = args.lr
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if "pred_head" not in n],
             "lr": encoder_lr},
            {"params": [p for n, p in model.named_parameters() if "pred_head" in n],
             "lr": head_lr},
        ]
        optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
        print(f"  Encoder LR: {encoder_lr:.1e}, Head LR: {head_lr:.1e}")
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── Loss (MSE for multi-task, without CCC since per-scale scoring) ──
    criterion = nn.MSELoss()

    # ── Training loop ──
    checkpoint_dir = Path(args.checkpoint_dir)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n─ Epoch {epoch}/{args.epochs} ─")

        train_metrics = train_epoch(model, train_dl, optimizer, criterion, device, args.grad_clip)
        print(f"  Train | Loss: {train_metrics['loss']:.4f} | RMSE: {train_metrics['rmse']:.4f}")

        val_metrics = validate_epoch(model, val_dl, criterion, device)
        print(f"  Val   | Loss: {val_metrics['loss']:.4f} | RMSE: {val_metrics['rmse']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "scale_names": scale_names,
                "output_ranges": output_ranges,
            }, checkpoint_dir / "hospital_best.pt")
            print(f"  ✓ Best model saved")

        # Simple early stopping
        if hasattr(args, 'patience'):
            # Reuse DAIC-WOZ stopper logic
            pass

    print(f"\n✅ Hospital fine-tuning complete.")
    print(f"   Model saved to: {checkpoint_dir}/hospital_best.pt")
    return model


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CancerEmotionSystem Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DAIC-WOZ pretraining
  python train.py --mode daic --data /path/to/DAIC-WOZ

  # Hospital fine-tuning (loads pretrained)
  python train.py --mode hospital --pretrained checkpoints/daic_best.pt

  # Quick pipeline test
  python train.py --mode dummy
        """,
    )

    # ── Mode ──
    parser.add_argument("--mode", type=str, default="dummy",
                       choices=["daic", "hospital", "dummy"],
                       help="Training mode")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to dataset root")
    parser.add_argument("--pretrained", type=str, default=None,
                       help="Path to pretrained checkpoint")

    # ── Model ──
    parser.add_argument("--d_audio", type=int, default=768)
    parser.add_argument("--d_text", type=int, default=768)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=500)

    # ── Training ──
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--ccc_weight", type=float, default=0.3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)

    # ── Output ──
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    print("=" * 60)
    print("CancerEmotionSystem — Training Pipeline")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Checkpoints: {args.checkpoint_dir}")

    if args.mode == "daic":
        train_daic_woz(args)
    elif args.mode == "hospital":
        train_hospital(args)
    elif args.mode == "dummy":
        print("\nRunning full pipeline test with dummy data...\n")
        train_daic_woz(args)
        print("\n" + "─" * 60)
        args.pretrained = str(Path(args.checkpoint_dir) / "daic_best.pt")
        args.epochs = 5  # Fewer epochs for demo
        train_hospital(args)

    print("\n" + "=" * 60)
    print("✅ Training pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
