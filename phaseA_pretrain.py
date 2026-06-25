#!/usr/bin/env python3
"""
Phase A: Audio Emotion Pretraining
----------------------------------
Pretrain audio encoder on RAVDESS + CREMA-D emotion classification.
Uses wav2vec2 feature extractor + TemporalConv + Transformer → 6-class emotion.

Output: checkpoints/phaseA_emotion_best.pt (audio encoder weights)

Usage:
  python phaseA_pretrain.py
"""

import os, sys, json, argparse, time, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import random
import torch
import torch.nn as nn

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import soundfile as sf
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm

# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════

# 6 common emotions
EMOTION_MAP = {
    "neutral": 0, "happy": 1, "sad": 2,
    "angry": 3, "fearful": 4, "disgust": 5,
}
NUM_CLASSES = len(EMOTION_MAP)
EMOTION_NAMES = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]

# RAVDESS: filename format 03-01-XX-... where XX is emotion code
RAVDESS_EMO = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
               "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}

# CREMA-D: filename format XXXX_XXX_XX.wav where middle part is emotion
# e.g. 1001_DFA_ANG_XX.wav
CREMAD_EMO = {"NEU": "neutral", "HAP": "happy", "SAD": "sad",
              "ANG": "angry", "FEA": "fearful", "DIS": "disgust"}

# ═══════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════

class EmotionDataset(Dataset):
    """Load RAVDESS + CREMA-D WAV files with emotion labels."""
    
    def __init__(self, ravdess_root, cremad_root, sample_rate=16000, max_duration=8.0):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.samples = []
        
        # ── RAVDESS ──
        if ravdess_root and Path(ravdess_root).exists():
            for wav_path in Path(ravdess_root).rglob("*.wav"):
                fname = wav_path.stem
                parts = fname.split("-")
                if len(parts) >= 3:
                    emo_code = parts[2]
                    emo_name = RAVDESS_EMO.get(emo_code)
                    if emo_name and emo_name in EMOTION_MAP:
                        self.samples.append((str(wav_path), EMOTION_MAP[emo_name], "ravdess"))
        
        # ── CREMA-D ──
        if cremad_root and Path(cremad_root).exists():
            for wav_path in Path(cremad_root).glob("*.wav"):
                fname = wav_path.stem
                parts = fname.split("_")
                if len(parts) >= 3:
                    emo_code = parts[2]
                    emo_name = CREMAD_EMO.get(emo_code)
                    if emo_name and emo_name in EMOTION_MAP:
                        self.samples.append((str(wav_path), EMOTION_MAP[emo_name], "cremad"))
        
        print(f"Loaded {len(self.samples)} samples ({sum(1 for s in self.samples if s[2]=='ravdess')} RAVDESS, {sum(1 for s in self.samples if s[2]=='cremad')} CREMA-D)")
        self._print_distribution()
    
    def _print_distribution(self):
        dist = defaultdict(int)
        for _, label, _ in self.samples:
            dist[EMOTION_NAMES[label]] += 1
        print("Distribution:", dict(dist))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        waveform, sr = sf.read(path)
        
        # Convert to torch and mono
        waveform = torch.from_numpy(waveform).float()
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)  # stereo → mono
        
        # Resample if needed
        if sr != self.sample_rate:
            import torchaudio.functional as F
            waveform = F.resample(waveform.unsqueeze(0), sr, self.sample_rate).squeeze(0)
        
        # Trim or pad
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]
        elif waveform.shape[0] < self.max_samples:
            waveform = torch.cat([
                waveform,
                torch.zeros(self.max_samples - waveform.shape[0])
            ])
        
        return waveform, torch.tensor(label, dtype=torch.long)
    
    def get_sample_weights(self, ravdess_weight=7.0):
        """Return per-sample weights to balance RAVDESS vs CREMA-D."""
        weights = []
        for _, _, source in self.samples:
            weights.append(ravdess_weight if source == "ravdess" else 1.0)
        return weights


def collate_fn(batch):
    """Pad waveforms to same length."""
    waveforms, labels = zip(*batch)
    max_len = max(w.shape[0] for w in waveforms)
    
    padded = torch.zeros(len(waveforms), max_len)
    mask = torch.zeros(len(waveforms), max_len, dtype=torch.bool)
    for i, w in enumerate(waveforms):
        padded[i, :w.shape[0]] = w
        mask[i, :w.shape[0]] = True
    
    return padded, mask, torch.stack(labels)


# ═══════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════

class AudioEmotionEncoder(nn.Module):
    """
    Audio emotion encoder for pretraining.
    wav2vec2 → TemporalConv → Transformer → Classification head.
    """
    def __init__(self, num_classes=NUM_CLASSES, d_model=256, n_layers=4, n_heads=8, 
                 d_ff=1024, dropout=0.1, freeze_wav2vec2=True):
        super().__init__()
        
        # Wav2Vec2 feature extractor (frozen)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.d_audio = self.wav2vec2.config.hidden_size  # 768
        
        if freeze_wav2vec2:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        # Temporal convolution (reduce frame rate)
        self.tconv = nn.Sequential(
            nn.Conv1d(self.d_audio, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),
        )
        
        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=5000, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        
        self.d_model = d_model
    
    def forward(self, waveform, attention_mask=None):
        B = waveform.shape[0]
        
        # wav2vec2 feature extraction
        with torch.set_grad_enabled(not all(p.requires_grad == False for p in self.wav2vec2.parameters())):
            w2v_out = self.wav2vec2(waveform, attention_mask=attention_mask)
            features = w2v_out.last_hidden_state  # [B, T_w2v, 768]
        
        # Temporal conv: [B, T_w2v, 768] → [B, 768, T_w2v] → [B, d_model, T_conv]
        features = features.transpose(1, 2)  # [B, 768, T]
        features = self.tconv(features)       # [B, d_model, T']
        features = features.transpose(1, 2)   # [B, T', d_model]
        
        # Positional encoding
        features = self.pos_enc(features)
        
        # Transformer
        features = self.transformer(features)  # [B, T', d_model]
        
        # Pool: mean over time
        pooled = features.mean(dim=1)  # [B, d_model]
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits, pooled


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


# ═══════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    losses, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"Train E{epoch}/{total_epochs}")
    
    for waveform, mask, labels in pbar:
        waveform = waveform.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(waveform, attention_mask=mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{100*correct/total:.1f}%"})
    
    return losses / len(loader), correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    losses, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    for waveform, mask, labels in tqdm(loader, desc="Val"):
        waveform = waveform.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        
        logits, _ = model(waveform, attention_mask=mask)
        loss = criterion(logits, labels)
        
        losses += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    # Per-class accuracy
    per_class = {}
    for cls_name, cls_id in EMOTION_MAP.items():
        mask_cls = torch.tensor(all_labels) == cls_id
        if mask_cls.sum() > 0:
            cls_correct = (torch.tensor(all_preds)[mask_cls] == cls_id).sum().item()
            per_class[cls_name] = cls_correct / mask_cls.sum().item()
    
    return losses / len(loader), correct / total, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ravdess", type=str, default="/data/disk1/datasets/ravdess")
    parser.add_argument("--cremad", type=str, default="/data/disk1/datasets/CREMA-D-master/AudioWAV")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phaseA")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ravdess_weight", type=float, default=7.0,
                        help="Weight multiplier for RAVDESS samples (default 7.0 = 7442/1056)")
    parser.add_argument("--freeze_wav2vec2", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ── Data ──
    print("\n═══ Loading Data ═══")
    dataset = EmotionDataset(args.ravdess, args.cremad)
    
    # Split
    n_total = len(dataset)
    n_val = max(int(n_total * 0.15), 32)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)
    
    # Weighted sampling to balance RAVDESS (few) vs CREMA-D (many)
    all_weights = dataset.get_sample_weights(ravdess_weight=args.ravdess_weight)
    train_weights = [all_weights[i] for i in train_ds.indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    print(f"  RAVDESS weight: {args.ravdess_weight}x | Effective samples/epoch: {len(train_weights)}")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    # ── Model ──
    print("\n═══ Building Model ═══")
    model = AudioEmotionEncoder(
        num_classes=NUM_CLASSES,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        freeze_wav2vec2=args.freeze_wav2vec2,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,} total, {n_trainable:,} trainable")
    
    # ── Optimizer ──
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    # ── Training ──
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_epoch = 1
    best_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0)
        history = ckpt.get("history", history)
        print(f"Resumed at epoch {start_epoch}, best_acc={best_acc*100:.1f}%")
    
    print(f"\n═══ Training ({start_epoch} → {args.epochs}) ═══")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
        val_loss, val_acc, per_class = val_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"  Val Loss: {val_loss:.4f} | Acc: {100*val_acc:.1f}% | "
              f"Best: {100*best_acc:.1f}%")
        print(f"  Per-class: {', '.join(f'{k}:{100*v:.0f}%' for k,v in per_class.items())}")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": best_acc,
                "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            }, os.path.join(args.checkpoint_dir, "phaseA_best.pt"))
            print(f"  ✓ Saved best (acc={100*best_acc:.1f}%)")
        
        # Save periodic
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_acc,
                "best_acc": best_acc,
                "history": history,
            }, os.path.join(args.checkpoint_dir, f"phaseA_epoch{epoch}.pt"))
    
    # Final save
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_acc": val_acc,
        "best_acc": best_acc,
        "history": history,
    }, os.path.join(args.checkpoint_dir, "phaseA_final.pt"))
    with open(os.path.join(args.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Phase A complete! Best acc: {100*best_acc:.1f}%")
    print(f"   Model: {args.checkpoint_dir}/phaseA_best.pt")


if __name__ == "__main__":
    main()
