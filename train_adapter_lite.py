#!/usr/bin/env python3
"""Lightweight adapter: pre-extracted wav2vec2 features -> MLP adapter -> classifier. Fast & no OOM."""
import os, sys, json, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import soundfile as sf, pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE = Path('/data/disk1/CancerEmotionSystem/s2')
SR, MAX_SEC = 16000, 120
DEVICE = torch.device('cuda:0')
BATCH, EPOCHS, LR = 8, 50, 2e-4
SEEDS = [42, 142, 242]
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'

# ---- Step 1: Extract wav2vec2 features ----
def extract_features():
    from transformers import Wav2Vec2Model
    w2v = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base').to(DEVICE)
    w2v.eval()
    data_dir = Path('/data/disk1/datasets/modma/extracted/audio_lanzhou_2015')
    xlsx = data_dir / 'subjects_information_audio_lanzhou_2015.xlsx'
    df = pd.read_excel(xlsx)
    df.columns = ['sid','type','age','gender','edu','PHQ9','CTQ','LES','SSRS','GAD7','PSQI','_a','_b']
    df['sid'] = df['sid'].astype(str).str.zfill(8)
    
    features, labels, sids = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting'):
        sid = row['sid']
        subj_dir = data_dir / sid
        if not subj_dir.exists(): continue
        wavs = sorted(subj_dir.glob('*.wav'))
        if len(wavs) < 10: continue
        segs = []
        for w in wavs:
            try:
                audio, _ = sf.read(str(w)); segs.append(torch.FloatTensor(audio))
            except: pass
        audio = torch.cat(segs)[:int(MAX_SEC * SR)]
        if len(audio) < 16000: audio = torch.cat([audio, torch.zeros(16000 - len(audio))])
        # Process in chunks to avoid OOM
        chunk_size = SR * 30
        chunk_feats = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = w2v(chunk).last_hidden_state.mean(dim=1).cpu()
            chunk_feats.append(out)
        feat = torch.cat(chunk_feats, dim=1).mean(dim=1)  # mean over all chunks -> [768]
        features.append(feat.squeeze(0))
        labels.append(int(row['PHQ9'] >= 10))
        sids.append(sid)
    
    X = torch.stack(features).numpy()
    y = np.array(labels)
    print(f'Extracted {len(X)} subjects, dim={X.shape[1]}, depressed={y.sum()}')
    return X, y

# ---- Step 2: Adapter model ----
class LiteAdapter(nn.Module):
    def __init__(self, d_in=768, d_hidden=256, n_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )
    def forward(self, x): return self.net(x)

def train_epoch(model, loader, opt, crit):
    model.train(); tl, tc, tt = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(x), y); loss.backward()
        opt.step()
        tl += loss.item(); tc += (model(x).argmax(1)==y).sum().item(); tt += len(y)
    return tl/len(loader), tc/tt

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds, labels, probs = [], [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds.extend(logits.argmax(1).cpu().tolist()); labels.extend(y.cpu().tolist())
        probs.extend(torch.softmax(logits,1)[:,1].cpu().tolist())
    return {'acc': accuracy_score(labels, preds), 'f1': f1_score(labels, preds, zero_division=0),
            'auc': roc_auc_score(labels, probs) if len(set(labels))>1 else 0.5}

print('Step 1: Extracting features...')
X, y = extract_features()

all_results = []
for seed in SEEDS:
    set_seed = lambda s: (random.seed(s), np.random.seed(s), torch.manual_seed(s))
    set_seed(seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_results = []
    for fold, (tr, vl) in enumerate(skf.split(X, y)):
        X_tr = torch.FloatTensor(X[tr]); y_tr = torch.LongTensor(y[tr])
        X_vl = torch.FloatTensor(X[vl]); y_vl = torch.LongTensor(y[vl])
        tr_dl = DataLoader(TensorDataset(X_tr, y_tr), BATCH, shuffle=True, drop_last=True)
        vl_dl = DataLoader(TensorDataset(X_vl, y_vl), BATCH)
        
        model = LiteAdapter().to(DEVICE)
        n_pos = y[tr].sum(); n_neg = len(tr) - n_pos
        crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, n_neg/max(n_pos,1)], device=DEVICE))
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        
        best_f1, best_state, patience = 0, None, 0
        for ep in range(1, EPOCHS+1):
            tl, ta = train_epoch(model, tr_dl, opt, crit)
            vm = evaluate(model, vl_dl)
            if vm['f1'] > best_f1: best_f1 = vm['f1']; patience = 0; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            else: patience += 1
            if ep % 10 == 0: print(f'  Fold{fold+1} Ep{ep}: loss={tl:.3f} f1={vm["f1"]:.3f} auc={vm["auc"]:.3f}')
            if patience >= 15: break
        
        model.load_state_dict(best_state)
        vm = evaluate(model, vl_dl)
        fold_results.append(vm)
        print(f'  Fold{fold+1}: f1={vm["f1"]:.3f} auc={vm["auc"]:.3f}')
    
    avg = {k: float(np.mean([f[k] for f in fold_results])) for k in fold_results[0]}
    std = {k: float(np.std([f[k] for f in fold_results])) for k in fold_results[0]}
    all_results.append({'seed': seed, 'avg': avg, 'std': std})
    print(f'Seed {seed}: Acc={avg["acc"]:.3f}+-{std["acc"]:.3f} F1={avg["f1"]:.3f}+-{std["f1"]:.3f} AUC={avg["auc"]:.3f}+-{std["auc"]:.3f}')

total = {k: float(np.mean([r['avg'][k] for r in all_results])) for k in all_results[0]['avg']}
tstd = {k: float(np.std([r['avg'][k] for r in all_results])) for k in all_results[0]['avg']}
print(f'\n=== MODMA Adapter Final ({len(SEEDS)}x5CV) ===')
print(f'Acc: {total["acc"]:.3f}+-{tstd["acc"]:.3f}  F1: {total["f1"]:.3f}+-{tstd["f1"]:.3f}  AUC: {total["auc"]:.3f}+-{tstd["auc"]:.3f}')
json.dump({'results': all_results, 'total': total, 'total_std': tstd}, open('results/adapter_modma.json', 'w'), indent=2)
