#!/usr/bin/env python3
"""Stage 1: Pre-train adapters on MODMA (Mandarin depression dataset)."""
import os, sys, json, random, argparse, warnings
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import soundfile as sf, pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
BASE = Path('/data/disk1/CancerEmotionSystem')
sys.path.insert(0, str(BASE))
from src.models.adapter_model import CrossCulturalModel

# Config
BATCH, EPOCHS, LR = 4, 30, 2e-4
SR, MAX_SEC = 16000, 120
DEVICE = torch.device('cpu')  # CPU to avoid OOM
SEEDS = [42, 142, 242]

class MODMADataset(Dataset):
    def __init__(self, data_dir, xlsx_path):
        self.data_dir = Path(data_dir)
        df = pd.read_excel(xlsx_path)
        df.columns = ['sid','type','age','gender','edu','PHQ9','CTQ','LES','SSRS','GAD7','PSQI','_a','_b']
        df['sid'] = df['sid'].astype(str).str.zfill(8)
        self.labels = df.set_index('sid')
        self.samples = []
        for sid in self.labels.index:
            subj_dir = self.data_dir / sid
            if not subj_dir.exists(): continue
            wavs = sorted(subj_dir.glob('*.wav'))
            if len(wavs) < 10: continue
            self.samples.append({'sid': sid, 'wavs': wavs})
        print(f'MODMA: {len(self.samples)} subjects')

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]; row = self.labels.loc[s['sid']]
        label = int(row['PHQ9'] >= 10)
        segs = []
        for w in s['wavs']:
            try:
                audio, _ = sf.read(str(w)); segs.append(torch.FloatTensor(audio))
            except: pass
        if not segs: segs = [torch.zeros(SR)]
        audio = torch.cat(segs)
        max_s = int(MAX_SEC * SR)
        if len(audio) > max_s: audio = audio[:max_s]
        else: audio = torch.cat([audio, torch.zeros(max_s - len(audio))])
        return audio, label, s['sid']

def collate(batch):
    audios = [b[0] for b in batch]; labels = [b[1] for b in batch]
    max_l = max(a.shape[0] for a in audios)
    padded = torch.zeros(len(audios), max_l)
    mask = torch.zeros(len(audios), max_l)
    for i, a in enumerate(audios):
        padded[i, :a.shape[0]] = a; mask[i, :a.shape[0]] = 1
    return padded, mask, torch.LongTensor(labels)

def train_epoch(model, loader, opt, crit):
    model.train(); tl, tc, tt = 0, 0, 0
    for audio, mask, labels in loader:
        audio, mask, labels = audio.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        logits = model(audio, [''] * len(labels), attention_mask=mask)
        loss = crit(logits, labels); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item(); tc += (logits.argmax(1)==labels).sum().item(); tt += len(labels)
    return tl/len(loader), tc/tt

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds, labels, probs = [], [], []
    for audio, mask, lbs in loader:
        audio, mask, lbs = audio.to(DEVICE), mask.to(DEVICE), lbs.to(DEVICE)
        logits = model(audio, [''] * len(lbs), attention_mask=mask)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(lbs.cpu().tolist())
        probs.extend(torch.softmax(logits,1)[:,1].cpu().tolist())
    return {
        'acc': accuracy_score(labels, preds), 'f1': f1_score(labels, preds, zero_division=0),
        'auc': roc_auc_score(labels, probs) if len(set(labels))>1 else 0.5,
    }

def run_fold(fold, train_idx, val_idx, dataset, seed):
    set_seed(seed)
    tr_ds = torch.utils.data.Subset(dataset, train_idx)
    vl_ds = torch.utils.data.Subset(dataset, val_idx)
    tr_dl = DataLoader(tr_ds, BATCH, shuffle=True, collate_fn=collate, drop_last=True)
    vl_dl = DataLoader(vl_ds, BATCH, collate_fn=collate)
    
    model = CrossCulturalModel(n_classes=2).to(DEVICE)
    print(f'  Fold{fold}: trainable={model.trainable_params():,} / total={model.total_params():,}')
    
    # Class weight
    lbs = [dataset[i][1] for i in train_idx]
    n_pos = sum(lbs); n_neg = len(lbs) - n_pos
    pos_w = n_neg / max(n_pos, 1)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    best_f1, best_state, patience = 0, None, 0
    for ep in range(1, EPOCHS+1):
        tl, ta = train_epoch(model, tr_dl, opt, crit)
        vm = evaluate(model, vl_dl)
        if vm['f1'] > best_f1: best_f1 = vm['f1']; patience = 0; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        else: patience += 1
        if ep % 10 == 0: print(f'    Ep{ep}: loss={tl:.3f} tr_acc={ta:.3f} val_f1={vm["f1"]:.3f} val_auc={vm["auc"]:.3f}')
        if patience >= 10: break
    
    model.load_state_dict(best_state)
    return evaluate(model, vl_dl), best_state, model

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='checkpoints/adapter_modma')
    args = parser.parse_args()
    global DEVICE; DEVICE = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)
    
    ds = MODMADataset('/data/disk1/datasets/modma/extracted/audio_lanzhou_2015',
                       '/data/disk1/datasets/modma/extracted/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx')
    
    labels = [ds[i][1] for i in range(len(ds))]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = []
    for seed in SEEDS:
        print(f'\n=== Seed {seed} ===')
        fold_results = []
        for fold, (tr, vl) in enumerate(skf.split(range(len(ds)), labels)):
            r, state, model = run_fold(fold+1, tr, vl, ds, seed)
            fold_results.append(r)
            if seed == SEEDS[0] and fold == 0:
                torch.save({'model_state_dict': state, 'config': {'d_model': 256, 'n_classes': 2}},
                          f'{args.output}/adapter_best.pt')
        avg = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
        std = {k: np.std([f[k] for f in fold_results]) for k in fold_results[0]}
        print(f'  Seed{seed} CV: Acc={avg["acc"]:.3f}±{std["acc"]:.3f} F1={avg["f1"]:.3f}±{std["f1"]:.3f} AUC={avg["auc"]:.3f}±{std["auc"]:.3f}')
        all_results.append({'seed': seed, 'avg': {k:float(v) for k,v in avg.items()}, 'std': {k:float(v) for k,v in std.items()}})
    
    total_avg = {k: float(np.mean([r['avg'][k] for r in all_results])) for k in all_results[0]['avg']}
    total_std = {k: float(np.std([r['avg'][k] for r in all_results])) for k in all_results[0]['avg']}
    print(f'\n=== MODMA Final ({len(SEEDS)} seeds x 5 folds) ===')
    print(f'  Acc: {total_avg["acc"]:.3f} ± {total_avg["acc"]:.3f}')
    print(f'  F1:  {total_avg["f1"]:.3f} ± {total_std["f1"]:.3f}')
    print(f'  AUC: {total_avg["auc"]:.3f} ± {total_std["auc"]:.3f}')
    
    with open(f'{args.output}/results.json', 'w') as f:
        json.dump({'individual': all_results, 'total_avg': total_avg, 'total_std': total_std}, f, indent=2)

if __name__ == '__main__':
    main()
