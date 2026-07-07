#!/usr/bin/env python3
"""Huang 2024 改进复现 v2 — 类别加权 + Warmup + 增强MLP + Early Stopping"""
import os, sys, json, random, warnings, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import soundfile as sf
from transformers import Wav2Vec2Model

warnings.filterwarnings('ignore')

DATA_DIR = '/data/disk1/datasets/diac_woz'
SEED_BASE = 103
PHQ_THRESHOLD = 10
MAX_SEC = 8
SR = 16000
BATCH = 8
EPOCHS = 30
LR_PEAK = 2e-5
WARMUP = 3
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N_RUNS = 3

def load_data():
    import pandas as pd
    labels = pd.read_csv(f'{DATA_DIR}/labels/train_split.csv').set_index('Participant_ID')
    pid2label = {pid: int(row['PHQ_Score'] >= PHQ_THRESHOLD) for pid, row in labels.iterrows()}
    segments = []
    for d in sorted(os.listdir(DATA_DIR)):
        if not d.endswith('_P'): continue
        pid = int(d.replace('_P', ''))
        if pid not in pid2label: continue
        csv_f = f'{DATA_DIR}/{d}/{pid}_Transcript.csv'
        wav_f = f'{DATA_DIR}/{d}/{pid}_AUDIO.wav'
        if not os.path.exists(csv_f) or not os.path.exists(wav_f): continue
        segs = []
        with open(csv_f) as f:
            for line in f.readlines()[1:]:
                p = line.strip().split(',')
                if len(p) >= 4:
                    try:
                        s, e = float(p[0]), float(p[1])
                        if e > s > 0: segs.append((s, e))
                    except: pass
        if len(segs) < 5: continue
        audio, _ = sf.read(wav_f)
        for si in range(0, len(segs), 3):
            group = segs[si:si+3]
            chunks = []
            for s, e in group:
                a, b = int(s*SR), int(e*SR)
                if b > a: chunks.append(audio[a:b])
            if chunks:
                merged = np.concatenate(chunks)
                if len(merged) < SR * 0.5: continue
                segments.append({'pid': pid, 'audio': merged, 'label': pid2label[pid]})
    return segments

class SegDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        audio = self.data[i]['audio']
        max_s = int(MAX_SEC * SR)
        if len(audio) > max_s:
            start = random.randint(0, len(audio) - max_s)
            audio = audio[start:start+max_s]
        else:
            audio = np.pad(audio, (0, max_s - len(audio)))
        return torch.FloatTensor(audio), self.data[i]['label'], self.data[i]['pid']

def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), torch.LongTensor([b[1] for b in batch]), [b[2] for b in batch]

class HuangModel(nn.Module):
    def __init__(self, d_model=768, num_classes=2, dropout=0.3):
        super().__init__()
        self.w2v = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.w2v(x).last_hidden_state.mean(dim=1))

def train_epoch(model, loader, opt, crit):
    model.train(); tl, tc, tt = 0, 0, 0
    for audio, labels, _ in loader:
        audio, labels = audio.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(audio), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item()
        tc += (model(audio).argmax(1) == labels).sum().item() if False else 0  # recalc for accuracy
        tt += len(labels)
    return tl / len(loader), tc / tt

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds, labels, probs = [], [], []
    for audio, lbs, _ in loader:
        audio, lbs = audio.to(DEVICE), lbs.to(DEVICE)
        logits = model(audio)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(lbs.cpu().tolist())
        probs.extend(torch.softmax(logits, 1)[:,1].cpu().tolist())
    return {'acc': accuracy_score(labels, preds), 'f1': f1_score(labels, preds, zero_division=0),
            'auc': roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5}

def run_experiment(name, train_data, val_data, test_data, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ds_tr = SegDataset(train_data); ds_vl = SegDataset(val_data); ds_te = SegDataset(test_data)
    dl_tr = DataLoader(ds_tr, BATCH, shuffle=True, collate_fn=collate_fn, drop_last=True)
    dl_vl = DataLoader(ds_vl, BATCH, collate_fn=collate_fn)
    dl_te = DataLoader(ds_te, BATCH, collate_fn=collate_fn)
    labels_tr = [d['label'] for d in train_data]
    n_pos = sum(labels_tr); n_neg = len(labels_tr) - n_pos
    pos_w = n_neg / max(n_pos, 1)
    print(f'  {name}: N={len(train_data)} pos={n_pos} neg={n_neg} w={pos_w:.2f}')
    model = HuangModel().to(DEVICE)
    for p in model.w2v.feature_extractor.parameters(): p.requires_grad = False
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w]).to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=LR_PEAK, weight_decay=0.01)
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    sch = SequentialLR(opt, [LinearLR(opt, 0.1, WARMUP), CosineAnnealingLR(opt, EPOCHS-WARMUP)], [WARMUP])
    best_f1, best_state, patience = 0, None, 0
    for ep in range(1, EPOCHS+1):
        if ep == 6:
            for p in model.w2v.feature_extractor.parameters(): p.requires_grad = True
        tl, ta = train_epoch(model, dl_tr, opt, crit)
        vm = evaluate(model, dl_vl); sch.step()
        if vm['f1'] > best_f1: best_f1 = vm['f1']; patience = 0; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        else: patience += 1
        if ep % 5 == 0: print(f'    Ep{ep:2d}: loss={tl:.3f} val_f1={vm["f1"]:.3f} val_auc={vm["auc"]:.3f}')
        if patience >= 10: break
    model.load_state_dict(best_state); tm = evaluate(model, dl_te)
    print(f'    BEST: acc={tm["acc"]:.3f} f1={tm["f1"]:.3f} auc={tm["auc"]:.3f}')
    return tm

def main():
    print(f'Device: {DEVICE}')
    data = load_data()
    print(f'Loaded {len(data)} segments, {len(set(d["pid"] for d in data))} participants')
    pids = list(set(d['pid'] for d in data)); random.Random(SEED_BASE).shuffle(pids)
    n = len(pids)
    tr_pids = set(pids[:int(n*0.6)]); vl_pids = set(pids[int(n*0.6):int(n*0.8)]); te_pids = set(pids[int(n*0.8):])
    train_d = [d for d in data if d['pid'] in tr_pids]
    val_d = [d for d in data if d['pid'] in vl_pids]
    test_d = [d for d in data if d['pid'] in te_pids]
    print(f'Split: train={len(train_d)} val={len(val_d)} test={len(test_d)}')
    results = []
    for run in range(1, N_RUNS+1):
        print(f'\n=== Run {run}/{N_RUNS} ===')
        results.append(run_experiment(f'Run{run}', train_d, val_d, test_d, SEED_BASE+run))
    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    std = {k: np.std([r[k] for r in results]) for k in results[0]}
    print(f'\n=== Final (N={N_RUNS}) ===')
    print(f'  Acc: {avg["acc"]:.4f} +- {std["acc"]:.4f}')
    print(f'  F1:  {avg["f1"]:.4f} +- {std["f1"]:.4f}')
    print(f'  AUC: {avg["auc"]:.4f} +- {std["auc"]:.4f}')
    outdir = os.path.dirname(os.path.abspath(__file__))
    with open(f'{outdir}/results_v2.json', 'w') as f:
        json.dump({'individual': results, 'avg': {k:float(v) for k,v in avg.items()}, 'std': {k:float(v) for k,v in std.items()}}, f, indent=2)
    print('Saved results_v2.json')

if __na
