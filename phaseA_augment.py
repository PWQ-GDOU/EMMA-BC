#!/usr/bin/env python3
"""
Phase A v4: RAVDESS-Only Audio Emotion Pretraining WITH Data Augmentation
--------------------------------------------------------------------------
Pretrain audio encoder on RAVDESS with audio augmentations.
Augmentations: pitch shift, time stretch, noise injection (70% probability each sample).
Uses wav2vec2 feature extractor + TemporalConv + Transformer -> 6-class emotion.

Output: checkpoints/phaseA_v4/phaseA_best.pt
"""

import os, sys, json, argparse, time, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import soundfile as sf
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm

# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════

EMOTION_MAP = {
    "neutral": 0, "happy": 1, "sad": 2,
    "angry": 3, "fearful": 4, "disgust": 5,
}
NUM_CLASSES = len(EMOTION_MAP)
EMOTION_NAMES = list(EMOTION_MAP.keys())

RAVDESS_EMO = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
               "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}

# ═══════════════════════════════════════════════════════
# Data Augmentation
# ═══════════════════════════════════════════════════════

class AudioAugment:
    """Random audio augmentations for training only."""
    def __init__(self, sample_rate=16000, p=0.7):
        self.sr = sample_rate
        self.p = p
    
    def __call__(self, waveform):
        """Apply one of 4 augmentations with probability p."""
        if torch.rand(1).item() > self.p:
            return waveform
        
        choice = torch.randint(0, 4, (1,)).item()
        
        if choice == 0:  # Pitch shift ±4 semitones
            steps = torch.randint(-4, 5, (1,)).item()
            if steps != 0:
                result = torchaudio.functional.pitch_shift(
                    waveform.unsqueeze(0), self.sr, steps
                )
                waveform = (result[0] if isinstance(result, tuple) else result).squeeze(0)
        
        elif choice == 1:  # Time stretch 0.8x - 1.2x
            rate = 0.8 + torch.rand(1).item() * 0.4
            result = torchaudio.functional.speed(
                waveform.unsqueeze(0), self.sr, rate
            )
            waveform = (result[0] if isinstance(result, tuple) else result).squeeze(0)
        
        elif choice == 2:  # Gaussian noise
            noise_level = 0.001 + torch.rand(1).item() * 0.01
            waveform = waveform + torch.randn_like(waveform) * noise_level
        
        elif choice == 3:  # Pitch shift + noise combo
            steps = torch.randint(-3, 4, (1,)).item()
            if steps != 0:
                result = torchaudio.functional.pitch_shift(
                    waveform.unsqueeze(0), self.sr, steps
                )
                waveform = (result[0] if isinstance(result, tuple) else result).squeeze(0)
            noise_level = 0.001 + torch.rand(1).item() * 0.005
            waveform = waveform + torch.randn_like(waveform) * noise_level
        
        return waveform

# ═══════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════

class EmotionDataset(Dataset):
    """Load RAVDESS WAV files with emotion labels. Optional augmentation."""
    
    def __init__(self, ravdess_root, sample_rate=16000, max_duration=8.0, augment=False):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.augment = AudioAugment(sample_rate) if augment else None
        self.samples = []
        
        if ravdess_root and Path(ravdess_root).exists():
            for wav_path in Path(ravdess_root).rglob("*.wav"):
                fname = wav_path.stem
                parts = fname.split("-")
                if len(parts) >= 3:
                    emo_code = parts[2]
                    emo_name = RAVDESS_EMO.get(emo_code)
                    if emo_name and emo_name in EMOTION_MAP:
                        self.samples.append((str(wav_path), EMOTION_MAP[emo_name]))
        
        print(f"Loaded {len(self.samples)} samples from RAVDESS")
        self._print_distribution()
    
    def _print_distribution(self):
        dist = defaultdict(int)
        for _, label in self.samples:
            dist[EMOTION_NAMES[label]] += 1
        print("Distribution:", dict(dist))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, sr = sf.read(path)
        
        waveform = torch.from_numpy(waveform).float()
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0), sr, self.sample_rate
            ).squeeze(0)
        
        # Apply augmentation (only during training)
        if self.augment is not None:
            waveform = self.augment(waveform)
        
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]
        
        return waveform, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    waveforms, labels = zip(*batch)
    max_len = max(w.shape[0] for w in waveforms)
    padded = torch.zeros(len(waveforms), max_len)
    mask = torch.zeros(len(waveforms), max_len, dtype=torch.bool)
    for i, w in enumerate(waveforms):
        padded[i, :w.shape[0]] = w
        mask[i, :w.shape[0]] = True
    return padded, mask, torch.stack(labels)


# ═══════════════════════════════════════════════════════
# Model (same as original)
# ═══════════════════════════════════════════════════════

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


class AudioEmotionEncoder(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, d_model=256, n_layers=4, n_heads=8,
                 d_ff=1024, dropout=0.1, freeze_wav2vec2=True):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.d_audio = self.wav2vec2.config.hidden_size
        if freeze_wav2vec2:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        self.tconv = nn.Sequential(
            nn.Conv1d(self.d_audio, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=5000, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.d_model = d_model
    
    def forward(self, waveform, attention_mask=None):
        with torch.set_grad_enabled(not all(p.requires_grad == False for p in self.wav2vec2.parameters())):
            w2v_out = self.wav2vec2(waveform, attention_mask=attention_mask)
            features = w2v_out.last_hidden_state
        
        features = features.transpose(1, 2)
        features = self.tconv(features)
        features = features.transpose(1, 2)
        features = self.pos_enc(features)
        features = self.transformer(features)
        pooled = features.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, pooled


# ═══════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    losses, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"Train E{epoch}/{total_epochs}")
    for waveform, mask, labels in pbar:
        waveform, mask, labels = waveform.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(waveform, attention_mask=mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{100*correct/total:.1f}%"})
    return losses / len(loader), correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    losses, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for waveform, mask, labels in tqdm(loader, desc="Val"):
        waveform, mask, labels = waveform.to(device), mask.to(device), labels.to(device)
        logits, _ = model(waveform, attention_mask=mask)
        loss = criterion(logits, labels)
        losses += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phaseA_v4")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_wav2vec2", action="store_true", default=True)
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable audio augmentation")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Augmentation: {'ON' if args.augment else 'OFF'}")
    print(f"Epochs: {args.epochs}")
    
    # ── Data ──
    print("\n═══ Loading Data ═══")
    dataset = EmotionDataset(args.ravdess, augment=args.augment)
    
    n_total = len(dataset)
    n_val = max(int(n_total * 0.15), 32)
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    print(f"Train: {n_train}, Val: {n_val}")
    
    # ── Model ──
    print("\n═══ Building Model ═══")
    model = AudioEmotionEncoder(
        num_classes=NUM_CLASSES, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
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
    best_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print(f"\n═══ Training ({args.epochs} epochs) ═══")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
        val_loss, val_acc, per_class = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"  Val Loss: {val_loss:.4f} | Acc: {100*val_acc:.1f}% | Best: {100*best_acc:.1f}%")
        print(f"  Per-class: {', '.join(f'{k}:{100*v:.0f}%' for k,v in per_class.items())}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_acc": best_acc,
                "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            }, os.path.join(args.checkpoint_dir, "phaseA_best.pt"))
            print(f"  \u2713 Saved best (acc={100*best_acc:.1f}%)")
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"phaseA_epoch{epoch}.pt"))
    
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "phaseA_final.pt"))
    with open(os.path.join(args.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n\u2705 Phase A v4 complete! Best acc: {100*best_acc:.1f}%")
    print(f"   Model: {args.checkpoint_dir}/phaseA_best.pt")


if __name__ == "__main__":
    main()
