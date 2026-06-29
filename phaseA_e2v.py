"""
Phase A Audio Emotion Encoder Pretraining — emotion2vec+ large edition.
Replaces wav2vec2-base with emotion2vec/emotion2vec_plus_large for SOTA emotion features.
"""
import os, sys, random, json, argparse, warnings
import numpy as np
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Config ──
EMOTIONS = ["neutral","happy","sad","angry","fearful","disgust"]
EMOTION_MAP = {e: i for i, e in enumerate(EMOTIONS)}
N_CLASSES = len(EMOTIONS)
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 30
D_MODEL = 1024  # emotion2vec+ large output dim
NHEAD = 8
N_LAYERS = 4
DROPOUT = 0.2
SEED = 42

# RAVDESS filename: 03-01-XX-... where XX = emotion code
RAVDESS_EMO = {"01":"neutral","02":"calm","03":"happy","04":"sad",
               "05":"angry","06":"fearful","07":"disgust","08":"surprised"}
# We only use 6 classes (exclude calm & surprised for consistency)

set_seed = lambda s=SEED: [random.seed(s), np.random.seed(s), torch.manual_seed(s),
                            torch.cuda.manual_seed_all(s)] or torch.backends.cudnn.deterministic.__setattr__('__call__', lambda _: True)


# ── Dataset ──
class EmotionDataset(Dataset):
    def __init__(self, ravdess_root, max_duration=8.0, sr=16000):
        import torchaudio
        self.samples = []
        if Path(ravdess_root).exists():
            for wav_path in Path(ravdess_root).rglob("*.wav"):
                fn = wav_path.name
                parts = fn.split("-")
                if len(parts) >= 3:
                    emo_code = parts[2]
                    emo_name = RAVDESS_EMO.get(emo_code)
                    if emo_name and emo_name in EMOTION_MAP:
                        label = EMOTION_MAP[emo_name]
                        self.samples.append((str(wav_path), label))
        self.max_len = int(max_duration * sr)
        print(f"Loaded {len(self.samples)} RAVDESS samples")
        from collections import Counter
        dist = Counter(EMOTIONS[s[1]] for s in self.samples)
        print(f"Distribution: {dict(dist)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torchaudio
        path, label = self.samples[idx]
        try:
            wav, native_sr = torchaudio.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read audio: {path}") from e
        if native_sr != 16000:
            wav = torchaudio.transforms.Resample(native_sr, 16000)(wav)
        wav = wav.mean(dim=0)[:self.max_len]  # mono, truncate
        return wav, label


def collate_fn(batch):
    max_len = max(w.shape[0] for w, _ in batch)
    B = len(batch)
    padded = torch.zeros(B, max_len)
    labels = torch.zeros(B, dtype=torch.long)
    mask = torch.zeros(B, max_len)
    for i, (w, l) in enumerate(batch):
        L = w.shape[0]
        padded[i, :L] = w
        mask[i, :L] = 1.0
        labels[i] = l
    return padded, mask, labels


# ── Model ──
class EmotionEncoder(nn.Module):
    """emotion2vec+ large → TemporalConv → Transformer → classifier"""
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.d_model = d_model
        
        # emotion2vec loaded separately (not as submodule, to avoid HF serialization issues)
        self.e2v = None
        
        # Temporal conv: downsample 4x
        self.tconv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        
        # Positional encoding
        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classifier
        self.classifier = nn.Linear(d_model, N_CLASSES)

    def forward(self, waveform, attention_mask):
        # emotion2vec: input [B, T_raw], output [B, T_feat, d_model]
        with torch.no_grad():
            result = self.e2v(waveform, output_dir=None)
            # result is dict with keys like 'feats', 'scores'
            features = result['feats']  # [B, T, d_model]
        
        # Conv: [B, T, d] → [B, d, T] → [B, d, T'] → [B, T', d]
        features = features.transpose(1, 2)
        features = self.tconv(features)
        features = features.transpose(1, 2)
        
        # Positional encoding
        features = self.pos_enc(features)
        
        # Transformer
        features = self.transformer(features)
        
        # Masked mean pool
        if attention_mask is not None:
            ds_factor = max(1, attention_mask.shape[1] // features.shape[1])
            mask_ds = attention_mask[:, ::ds_factor][:, :features.shape[1]]
            mask_f = mask_ds.unsqueeze(-1)
            pooled = (features * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = features.mean(dim=1)
        
        return self.classifier(pooled), pooled


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.shape[1]])


# ── Training ──
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"Train E{epoch}/{EPOCHS}")
    for waveform, mask, labels in pbar:
        waveform, mask, labels = waveform.to(device), mask.to(device), labels.to(device)
        logits, _ = model(waveform, mask)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total*100:.1f}%")
    return total_loss / len(loader), correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    class_correct, class_total = [0]*N_CLASSES, [0]*N_CLASSES
    for waveform, mask, labels in tqdm(loader, desc="Val"):
        waveform, mask, labels = waveform.to(device), mask.to(device), labels.to(device)
        logits, _ = model(waveform, mask)
        total_loss += criterion(logits, labels).item()
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for i in range(N_CLASSES):
            mask_i = labels == i
            class_correct[i] += (preds[mask_i] == i).sum().item()
            class_total[i] += mask_i.sum().item()
    acc = correct / total
    per_class = {e: f"{class_correct[i]/max(1,class_total[i])*100:.0f}%" for i,e in enumerate(EMOTIONS)}
    return total_loss / len(loader), acc, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ravdess", type=str, default="/data/disk1/datasets/ravdess")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phaseA_e2v")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    set_seed()
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Load emotion2vec ──
    print("\n>>> Loading emotion2vec+ large...")
    from funasr import AutoModel
    e2v = AutoModel(model="iic/emotion2vec_plus_large", trust_remote_code=True, device=args.device)
    print(">>> emotion2vec+ large loaded")

    # ── Data ──
    print("\n>>> Loading Data...")
    dataset = EmotionDataset(args.ravdess)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    # ── Model ──
    print("\n>>> Building Model...")
    model = EmotionEncoder(d_model=D_MODEL)
    model.e2v = e2v
    model.to(device)

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,} total, {trainable:,} trainable")

    # ── Optimizer ──
    # Only train classifier + transformer heads (emotion2vec is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_acc = 0.0
    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        history = ckpt.get("history", history)
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc*100:.1f}%")

    print(f"\n>>> Training ({start_epoch} -> {args.epochs})...\n")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc, per_class = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.1f}% | Best: {best_acc*100:.1f}%")
        print(f"  Per-class: {per_class}")

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            ckpt_path = f"{args.checkpoint_dir}/phaseA_e2v_best.pt"
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_acc": best_acc, "history": history
            }, ckpt_path)
            print(f"  >> Saved best (acc={best_acc*100:.1f}%)")

        # Save periodic
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_acc": best_acc, "history": history
            }, f"{args.checkpoint_dir}/phaseA_e2v_epoch{epoch}.pt")

    # Final
    torch.save({
        "epoch": args.epochs, "model": model.state_dict(),
        "best_acc": best_acc, "history": history
    }, f"{args.checkpoint_dir}/phaseA_e2v_final.pt")
    print(f"\n>>> Training done! Best val acc: {best_acc*100:.1f}%")


if __name__ == "__main__":
    main()
