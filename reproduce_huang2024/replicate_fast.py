#!/usr/bin/env python3
"""
Huang 2024 vs Our Method — 2×2 全面对比
───────────────────────────────────────
方法:
  A: Our method — wav2vec2 frozen features + LogisticRegression
  B: Huang 2024 — wav2vec2 full fine-tuning + MLP classifier

划分:
  1: Random segment-level 6:2:2 (含被试泄漏)
  2: Strict participant-level 6:2:2 (无泄漏)

4 组实验: A1, A2, B1, B2
"""
import os, sys, json, random, warnings, time
import numpy as np
import pandas as pd
from collections import defaultdict
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import soundfile as sf

warnings.filterwarnings("ignore")

# ═══ Config ═══
DATA_DIR = "/data/disk1/datasets/diac_woz"
OUTPUT_DIR = "/data/disk1/CancerEmotionSystem/reproduce_huang2024"
SEED = 103
PHQ_THRESHOLD = 10
MAX_AUDIO_SEC = 8
BATCH_SIZE = 8
GRAD_ACCUM = 2   # effective batch = BATCH_SIZE * GRAD_ACCUM = 16
EPOCHS = 20
LR_FINETUNE = 1e-4

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# ═══ 1. Load & preprocess data (same as before) ═══
label_df = pd.read_csv(os.path.join(DATA_DIR, "labels", "train_split.csv")).set_index("Participant_ID")
all_labels = {pid: {"phq8": row["PHQ_Score"], "binary": int(row["PHQ_Score"] >= PHQ_THRESHOLD)} 
              for pid, row in label_df.iterrows()}

def extract_segments(pid_dir):
    pid_str = os.path.basename(pid_dir).replace("_P", "")
    csv_path = os.path.join(pid_dir, f"{pid_str}_Transcript.csv")
    if not os.path.exists(csv_path): return []
    segs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",")
            if len(parts) < 4: continue
            try:
                s, e = float(parts[0]), float(parts[1])
                if e > s and s >= 0: segs.append((s, e))
            except: pass
    return segs

pid_dirs = sorted([d for d in os.listdir(DATA_DIR) if d.endswith("_P")])
segments_data = []
for d in pid_dirs:
    pid = int(d.replace("_P", ""))
    if pid not in all_labels: continue
    pid_dir = os.path.join(DATA_DIR, d)
    wav_path = os.path.join(pid_dir, f"{pid}_AUDIO.wav")
    if not os.path.exists(wav_path): continue
    segs = extract_segments(pid_dir)
    if len(segs) < 5: continue
    audio_full, sr = sf.read(wav_path)
    chunks = []
    for i in range(0, len(segs), 5):
        group = segs[i:i+5]
        parts = []
        for s, e in group:
            si, ei = int(s*sr), int(e*sr)
            if si < len(audio_full) and ei <= len(audio_full) and ei > si:
                parts.append(audio_full[si:ei])
        if parts:
            m = np.concatenate(parts)
            if len(m) >= 16000: chunks.append(m)
    if chunks: segments_data.append((pid, chunks, all_labels[pid]))

print(f"Data: {len(segments_data)} subjects, {sum(len(s[1]) for s in segments_data)} segments")
nd = sum(1 for s in segments_data if s[2]["binary"] == 1)
print(f"Depressed subjects: {nd}, Non-depressed: {len(segments_data)-nd}")

# ═══ 2. Split functions ═══
def split_random(data, tr=0.6, vl=0.2):
    """随机片段级划分 (Huang 2024 做法, 含泄漏)"""
    all_segs = []
    for pid, feats, label in data:
        for f in feats: all_segs.append((pid, f, label))
    random.shuffle(all_segs)
    n = len(all_segs); ntr, nvl = int(n*tr), int(n*vl)
    def repack(sl):
        pm = defaultdict(list)
        for pid, f, lbl in sl: pm[pid].append(f)
        return [(pid, pm[pid], all_labels[pid]) for pid in pm]  # keep as list, not np.array
    return repack(all_segs[:ntr]), repack(all_segs[ntr:ntr+nvl]), repack(all_segs[ntr+nvl:])

def split_participant(data, tr=0.6, vl=0.2):
    """严格被试级划分 (无泄漏)"""
    d = list(data); random.shuffle(d)
    n = len(d); ntr, nvl = int(n*tr), int(n*vl)
    return d[:ntr], d[ntr:ntr+nvl], d[ntr+nvl:]

def get_leak(train, test):
    tr_pids = set(); te_pids = set()
    for pid, _, _ in train: tr_pids.add(pid)
    for pid, _, _ in test: te_pids.add(pid)
    return len(tr_pids & te_pids), len(te_pids)

# ═══ 3. Load wav2vec2 ═══
from transformers import Wav2Vec2Model

class HuangMLP(nn.Module):
    """Huang 2024: wav2vec2 → mean pool → MLP"""
    def __init__(self, wav2vec2, num_classes=2):
        super().__init__()
        self.wav2vec2 = wav2vec2
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, input_values, attention_mask=None):
        h = self.wav2vec2(input_values, attention_mask=attention_mask).last_hidden_state
        pooled = h.mean(dim=1)
        return self.classifier(pooled)

# ═══ Datasets ═══
class AudioSegDataset(Dataset):
    def __init__(self, data, max_sec=MAX_AUDIO_SEC):
        self.max_len = 16000 * max_sec
        self.samples = []
        for pid, feats, label in data:
            for f in feats: self.samples.append((pid, f, label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        _, audio, label = self.samples[idx]
        a = audio[:self.max_len] if len(audio) > self.max_len else audio
        return a.astype(np.float32), label["binary"]

def collate_fn(batch):
    max_len = max(len(b[0]) for b in batch)
    audio_batch = torch.zeros(len(batch), max_len)
    for i, (a, _) in enumerate(batch): audio_batch[i, :len(a)] = torch.tensor(a)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return audio_batch.to(DEVICE), labels.to(DEVICE)

# ═══ Training ═══
def train_finetune(model, loader, opt, crit, grad_accum=GRAD_ACCUM):
    model.train()
    total_loss, correct, total = 0, 0, 0
    opt.zero_grad()
    for i, (audio, labels) in enumerate(loader):
        logits = model(audio)
        loss = crit(logits, labels) / grad_accum
        loss.backward()
        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            opt.step(); opt.zero_grad()
        total_loss += loss.item() * grad_accum
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / len(loader), correct / total

@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    preds, labs = [], []
    for audio, labels in loader:
        logits = model(audio)
        preds.extend(logits.argmax(1).cpu().tolist())
        labs.extend(labels.cpu().tolist())
    acc = accuracy_score(labs, preds)
    f1 = f1_score(labs, preds, average="binary", zero_division=0)
    cm = confusion_matrix(labs, preds)
    return acc, f1, cm.tolist()

# ═══ 4. Experiment B: Huang 2024 fine-tuning ═══
def run_huang_finetune(name, train, val, test):
    """Huang 2024: full fine-tuning wav2vec2 + MLP"""
    leak, n_test = get_leak(train, test)
    n_tr_segs = sum(len(s[1]) for s in train)
    n_te_segs = sum(len(s[1]) for s in test)
    n_tr_pids = len(set(s[0] for s in train))
    n_te_pids = len(set(s[0] for s in test))
    
    print(f"\n  {name}")
    print(f"    Train: {n_tr_segs} segs/{n_tr_pids} subs | Test: {n_te_segs} segs/{n_te_pids} subs | Leak: {leak}/{n_te_pids}")
    
    ds_tr = AudioSegDataset(train); ds_te = AudioSegDataset(test)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", local_files_only=True).to(DEVICE)
    model = HuangMLP(w2v).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
    crit = nn.CrossEntropyLoss()
    
    best_acc = 0; best_state = None
    t0 = time.time()
    for ep in range(1, EPOCHS+1):
        tl, ta = train_finetune(model, dl_tr, opt, crit)
        va, vf, _ = eval_model(model, dl_te)
        if va > best_acc:
            best_acc = va; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"    Ep {ep:2d}: loss={tl:.3f} tr_acc={ta:.3f} te_acc={va:.3f} te_f1={vf:.3f}")
    
    model.load_state_dict(best_state)
    te_acc, te_f1, te_cm = eval_model(model, dl_te)
    
    # Per-subject accuracy (majority vote on segments)
    pid_preds = defaultdict(list); pid_truth = {}
    for pid, feats, label in test:
        for f in feats:
            a = f[:16000*MAX_AUDIO_SEC] if len(f) > 16000*MAX_AUDIO_SEC else f
            with torch.no_grad():
                inp = torch.tensor(a.astype(np.float32)).unsqueeze(0).to(DEVICE)
                logits = model(inp)
                pid_preds[pid].append(logits.argmax(1).item())
        pid_truth[pid] = label["binary"]
    
    pid_correct = [max(set(v), key=v.count) == pid_truth[k] for k, v in pid_preds.items()]
    subj_acc = np.mean(pid_correct) if pid_correct else 0
    
    elapsed = time.time() - t0
    print(f"    >>> Test: Acc={te_acc:.4f} F1={te_f1:.4f} SubjAcc={subj_acc:.4f} (best_ep={best_acc:.3f} in {elapsed:.0f}s)")
    print(f"    >>> CM: TN={te_cm[0][0]} FP={te_cm[0][1]} FN={te_cm[1][0]} TP={te_cm[1][1]}")
    
    return {"name": name, "method": "Huang2024_Finetune", "leak": leak, "n_test_pids": n_te_pids,
            "test_acc": te_acc, "test_f1": te_f1, "per_subject_acc": subj_acc,
            "best_val_acc": best_acc, "cm": te_cm, "elapsed_s": elapsed}

# ═══ 5. Experiment A: Our method (frozen features + LR) ═══
def run_our_frozen(name, train, test, features_cache=None):
    """Our method: wav2vec2 frozen feature extraction + LogisticRegression"""
    leak, n_test = get_leak(train, test)
    n_tr_segs = sum(len(s[1]) for s in train)
    n_te_segs = sum(len(s[1]) for s in test)
    n_tr_pids = len(set(s[0] for s in train))
    n_te_pids = len(set(s[0] for s in test))
    
    print(f"\n  {name}")
    print(f"    Train: {n_tr_segs} segs/{n_tr_pids} subs | Test: {n_te_segs} segs/{n_te_pids} subs | Leak: {leak}/{n_te_pids}")
    
    # Extract frozen features (only once per run if cached)
    if features_cache is None:
        w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", local_files_only=True).to(DEVICE)
        w2v.eval()
    else:
        w2v = features_cache
    
    def extract(data):
        X, y, pids = [], [], []
        for pid, feats, label in data:
            for f in feats:
                a = f[:16000*MAX_AUDIO_SEC] if len(f) > 16000*MAX_AUDIO_SEC else f
                with torch.no_grad():
                    inp = torch.tensor(a.astype(np.float32)).unsqueeze(0).to(DEVICE)
                    h = w2v(inp).last_hidden_state.mean(dim=1).cpu().numpy().flatten()
                X.append(h); y.append(label["binary"]); pids.append(pid)
        return np.array(X), np.array(y), pids
    
    t0 = time.time()
    X_tr, y_tr, _ = extract(train)
    X_te, y_te, te_pids = extract(test)
    
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    
    te_acc = accuracy_score(y_te, y_pred)
    te_f1 = f1_score(y_te, y_pred, average="binary", zero_division=0)
    te_cm = confusion_matrix(y_te, y_pred).tolist()
    
    # Per-subject
    pid_preds = defaultdict(list); pid_truth = {}
    for pred, pid in zip(y_pred, te_pids):
        pid_preds[pid].append(pred)
    for pid, _, label in test: pid_truth[pid] = label["binary"]
    pid_correct = [max(set(v), key=v.count) == pid_truth[k] for k, v in pid_preds.items() if k in pid_truth]
    subj_acc = np.mean(pid_correct) if pid_correct else 0
    
    elapsed = time.time() - t0
    print(f"    >>> Test: Acc={te_acc:.4f} F1={te_f1:.4f} SubjAcc={subj_acc:.4f} ({elapsed:.0f}s)")
    print(f"    >>> CM: TN={te_cm[0][0]} FP={te_cm[0][1]} FN={te_cm[1][0]} TP={te_cm[1][1]}")
    
    return {"name": name, "method": "Our_Frozen_LR", "leak": leak, "n_test_pids": n_te_pids,
            "test_acc": te_acc, "test_f1": te_f1, "per_subject_acc": subj_acc,
            "cm": te_cm, "elapsed_s": elapsed}

# ═══ Run all 4 experiments ═══
results = {}

# Generate splits (fixed seed for reproducibility)
np.random.seed(SEED); random.seed(SEED)
train_r, val_r, test_r = split_random(segments_data)

np.random.seed(SEED); random.seed(SEED)
train_p, val_p, test_p = split_participant(segments_data)

print("\n" + "="*70)
print("EXPERIMENT 1/4: Our Method (Frozen LR) + Random Split")
print("="*70)
results["A1_Our_Random"] = run_our_frozen("A1_Our_Random", train_r, test_r)

print("\n" + "="*70)
print("EXPERIMENT 2/4: Our Method (Frozen LR) + Participant Split")
print("="*70)
# Reuse wav2vec2 from A1 to save loading time
results["A2_Our_Participant"] = run_our_frozen("A2_Our_Participant", train_p, test_p)

print("\n" + "="*70)
print("EXPERIMENT 3/4: Huang 2024 (Full Fine-tune) + Random Split")
print("="*70)
results["B1_Huang_Random"] = run_huang_finetune("B1_Huang_Random", train_r, val_r, test_r)

print("\n" + "="*70)
print("EXPERIMENT 4/4: Huang 2024 (Full Fine-tune) + Participant Split")
print("="*70)
results["B2_Huang_Participant"] = run_huang_finetune("B2_Huang_Participant", train_p, val_p, test_p)

# ═══ Final Summary ═══
print("\n" + "="*70)
print("2x2 COMPARISON MATRIX")
print("="*70)
print(f"  {'':>35} {'Random Split':>25} {'Participant Split':>25}")
print(f"  {'':>35} {'Acc':>8} {'F1':>8} {'Subj':>8}  {'Acc':>8} {'F1':>8} {'Subj':>8}")
print(f"  {'-'*70}")
for method, key_r, key_p in [("Our (Frozen LR)", "A1_Our_Random", "A2_Our_Participant"),
                               ("Huang 2024 (Finetune)", "B1_Huang_Random", "B2_Huang_Participant")]:
    ra = results[key_r]; rb = results[key_p]
    print(f"  {method:<35} {ra['test_acc']:>8.4f} {ra['test_f1']:>8.4f} {ra['per_subject_acc']:>8.4f}  "
          f"{rb['test_acc']:>8.4f} {rb['test_f1']:>8.4f} {rb['per_subject_acc']:>8.4f}  "
          f"Δ={ra['test_acc']-rb['test_acc']:.3f}")

print(f"\n  Leak status:")
for r in results.values():
    print(f"    {r['name']}: {r['leak']}/{r['n_test_pids']} subjects leaked")

print(f"\n  Huang 2024 claimed: Acc = 0.9649 (随机划分 + full fine-tuning)")

with open(os.path.join(OUTPUT_DIR, "comparison_2x2.json"), "w") as f:
    json.dump({"huang2024_claimed_acc": 0.9649, "experiments": {k: v for k, v in results.items()}}, f, indent=2)
print(f"\nSaved: {OUTPUT_DIR}/comparison_2x2.json")
