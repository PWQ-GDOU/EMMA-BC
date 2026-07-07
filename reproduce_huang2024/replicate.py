#!/usr/bin/env python3
"""
复现 Huang et al. 2024 — 基于实际服务器数据
─────────────────────────────────────────
两版对比:
  A: 片段级随机划分 6:2:2 (Huang 2024做法, 含被试泄漏)
  B: 严格被试级划分 6:2:2 (无泄漏)
"""
import os, sys, json, random, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import soundfile as sf

warnings.filterwarnings("ignore")

DATA_DIR = "/data/disk1/datasets/diac_woz"
OUTPUT_DIR = "/data/disk1/CancerEmotionSystem/reproduce_huang2024"
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
SEED = 103
PHQ_THRESHOLD = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ═══ Step 1: Load labels ═══
label_df = pd.read_csv(os.path.join(DATA_DIR, "labels", "train_split.csv"))
label_df = label_df.set_index("Participant_ID")
all_labels = {}
for pid, row in label_df.iterrows():
    all_labels[pid] = {"phq8": row["PHQ_Score"], "binary": int(row["PHQ_Score"] >= PHQ_THRESHOLD)}

print(f"Labels loaded: {len(all_labels)} subjects")
n_dep = sum(1 for v in all_labels.values() if v["binary"] == 1)
print(f"  Depressed: {n_dep}, Non-depressed: {len(all_labels)-n_dep}")

# ═══ Step 2: Extract audio segments ═══
def extract_segments(pid_dir):
    """从 Transcript.csv 提取语音片段起止时间"""
    pid_str = os.path.basename(pid_dir).replace("_P", "")
    csv_path = os.path.join(pid_dir, f"{pid_str}_Transcript.csv")
    if not os.path.exists(csv_path):
        return []
    
    segments = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        return []
    
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) < 4:
            continue
        try:
            start, end = float(parts[0]), float(parts[1])
            if end > start and start >= 0:
                segments.append((start, end))
        except ValueError:
            continue
    return segments

def merge_segments(wav_path, segments, group_size=5):
    """每 group_size 个片段合并为一个音频 (复现 Huang 2024)"""
    audio, sr = sf.read(wav_path)
    merged = []
    
    for i in range(0, len(segments), group_size):
        group = segments[i:i+group_size]
        chunks = []
        for start, end in group:
            s = int(start * sr)
            e = int(end * sr)
            if s < len(audio) and e <= len(audio) and e > s:
                chunks.append(audio[s:e])
        if chunks:
            merged_chunk = np.concatenate(chunks)
            if len(merged_chunk) >= 16000:  # at least 1 second
                merged.append(merged_chunk)
    
    return merged

pid_dirs = sorted([d for d in os.listdir(DATA_DIR) if d.endswith("_P")])
print(f"\nP_ID dirs: {len(pid_dirs)}")

all_samples = []
pids_ok = set()

for d in pid_dirs:
    pid = int(d.replace("_P", ""))
    if pid not in all_labels:
        continue
    
    pid_dir = os.path.join(DATA_DIR, d)
    wav_path = os.path.join(pid_dir, f"{pid}_AUDIO.wav")
    if not os.path.exists(wav_path):
        continue
    
    segments = extract_segments(pid_dir)
    if len(segments) < 5:
        continue
    
    merged = merge_segments(wav_path, segments, group_size=5)
    label = all_labels[pid]
    
    for audio_chunk in merged:
        all_samples.append((pid, audio_chunk, label))
    
    pids_ok.add(pid)

print(f"Samples: {len(all_samples)} from {len(pids_ok)} subjects")
print(f"Per subject: {len(all_samples)/len(pids_ok):.1f} segments")
print(f"Depressed segments: {sum(1 for s in all_samples if s[2]['binary']==1)}")

# ═══ Step 3: Model ═══
from transformers import Wav2Vec2Processor, Wav2Vec2Model

print("\nLoading wav2vec2-base...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

class DepressionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2),
        )
    
    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)  # mean pool over time
        return self.classifier(pooled)

class SegDataset(Dataset):
    def __init__(self, samples, max_sec=10):
        self.samples = samples
        self.max_len = 16000 * max_sec
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        _, audio, label = self.samples[idx]
        if len(audio) > self.max_len:
            audio = audio[:self.max_len]
        return audio.astype(np.float32), label["binary"]

def collate(batch):
    max_len = max(len(b[0]) for b in batch)
    audio_batch = np.zeros((len(batch), max_len), dtype=np.float32)
    for i, (a, _) in enumerate(batch):
        audio_batch[i, :len(a)] = a
    
    inputs = processor(audio_batch.tolist(), sampling_rate=16000, return_tensors="pt", padding=True)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    iv = inputs.input_values.to(device)
    am = inputs.get("attention_mask", torch.ones_like(iv)).to(device)
    return iv, am, labels.to(device)

# ═══ Training ═══
def train_epoch(model, loader, opt, crit):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for iv, am, labels in loader:
        opt.zero_grad()
        with torch.no_grad():
            h = wav2vec2(iv, attention_mask=am).last_hidden_state
        logits = model(h)
        loss = crit(logits, labels)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return loss_sum/len(loader), correct/total

def evaluate(model, loader):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for iv, am, labels in loader:
            h = wav2vec2(iv, attention_mask=am).last_hidden_state
            logits = model(h)
            preds.extend(logits.argmax(1).cpu().tolist())
            labs.extend(labels.cpu().tolist())
    acc = accuracy_score(labs, preds)
    f1 = f1_score(labs, preds, average='binary', zero_division=0)
    cm = confusion_matrix(labs, preds)
    return acc, f1, cm

# ═══ Splits ═══
def split_random(samples, train_r=0.6, val_r=0.2):
    indices = list(range(len(samples)))
    random.shuffle(indices)
    n = len(samples)
    n_tr = int(n * train_r)
    n_vl = int(n * val_r)
    return (
        [samples[i] for i in indices[:n_tr]],
        [samples[i] for i in indices[n_tr:n_tr+n_vl]],
        [samples[i] for i in indices[n_tr+n_vl:]],
    )

def split_participant(samples, train_r=0.6, val_r=0.2):
    pid_map = defaultdict(list)
    for s in samples:
        pid_map[s[0]].append(s)
    pids = list(pid_map.keys())
    random.shuffle(pids)
    n = len(pids)
    n_tr = int(n * train_r)
    n_vl = int(n * val_r)
    return (
        [s for pid in pids[:n_tr] for s in pid_map[pid]],
        [s for pid in pids[n_tr:n_tr+n_vl] for s in pid_map[pid]],
        [s for pid in pids[n_tr+n_vl:] for s in pid_map[pid]],
    )

def run_exp(name, train, val, test):
    tr_pids = set(s[0] for s in train)
    te_pids = set(s[0] for s in test)
    leak = len(tr_pids & te_pids)
    
    print(f"\n  {name}")
    print(f"    Train: {len(train)} samples, {len(tr_pids)} PIDs")
    print(f"    Val:   {len(val)} samples")
    print(f"    Test:  {len(test)} samples, {len(te_pids)} PIDs")
    print(f"    Leaked PIDs: {leak}")
    
    loader_tr = DataLoader(SegDataset(train), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    loader_vl = DataLoader(SegDataset(val), batch_size=BATCH_SIZE, collate_fn=collate)
    loader_te = DataLoader(SegDataset(test), batch_size=BATCH_SIZE, collate_fn=collate)
    
    model = DepressionClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    
    best_va = 0; best_state = None
    for ep in range(1, EPOCHS+1):
        tl, ta = train_epoch(model, loader_tr, opt, crit)
        va, vf, _ = evaluate(model, loader_vl)
        if va > best_va:
            best_va = va; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"    Ep {ep:2d}: tr_loss={tl:.3f} tr_acc={ta:.3f} va_acc={va:.3f} va_f1={vf:.3f}")
    
    model.load_state_dict(best_state)
    ta, tf, tcm = evaluate(model, loader_te)
    print(f"    >>> BEST va_acc={best_va:.3f} | TEST acc={ta:.4f} f1={tf:.4f}")
    print(f"    >>> CM: TN={tcm[0,0]} FP={tcm[0,1]} FN={tcm[1,0]} TP={tcm[1,1]}")
    
    return {"name": name, "leak": leak, "best_va": best_va, "test_acc": ta, "test_f1": tf, "cm": tcm.tolist()}

# ═══ Run ═══
print("\n" + "="*60)
print("PROTOCOL A: Huang 2024 — Random segment-level split (6:2:2, seed=103)")
print("="*60)
random.seed(SEED); np.random.seed(SEED)
ta, va, tea = split_random(all_samples)
ra = run_exp("A_RandomSplit_Huang2024", ta, va, tea)

print("\n" + "="*60)
print("PROTOCOL B: Strict participant-level split (6:2:2)")
print("="*60)
random.seed(SEED); np.random.seed(SEED)
tb, vb, teb = split_participant(all_samples)
rb = run_exp("B_ParticipantSplit_OurMethod", tb, vb, teb)

# ═══ Report ═══
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"  Huang 2024 claimed:  Acc = 0.9649")
print(f"  {'Protocol':<35} {'Leak':>6} {'Test Acc':>10} {'Test F1':>10}")
print(f"  {'-'*63}")
for r in [ra, rb]:
    print(f"  {r['name']:<35} {r['leak']:>6} {r['test_acc']:>10.4f} {r['test_f1']:>10.4f}")

with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump({"huang2024_claimed_acc": 0.9649, "protocol_a": ra, "protocol_b": rb}, f, indent=2)
print(f"\nSaved: {OUTPUT_DIR}/results.json")
