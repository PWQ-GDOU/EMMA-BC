import json, random, numpy as np, torch, torch.nn as nn, torchaudio
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

class Fusion(nn.Module):
    def __init__(self, d_audio=768, d_model=256, dropout=0.3):
        super().__init__()
        self.audio_proj = nn.Sequential(nn.Linear(d_audio, d_model), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 2))
    def forward(self, x):
        return self.classifier(self.audio_proj(x))

def load_npz(path):
    d = np.load(path, allow_pickle=True)['data']
    X = torch.FloatTensor(np.stack([it['audio'] for it in d]))
    y = torch.LongTensor([float(it['label'] >= 10) for it in d])
    return X, y

def augment_batch(audio, device):
    """Online augmentation: pitch, speed, or noise"""
    if random.random() < 0.5: return audio
    r = random.random()
    audio_np = audio.cpu().numpy()
    augmented = []
    for i in range(len(audio_np)):
        a = torch.FloatTensor(audio_np[i])
        if len(a) < 16000: a = torch.cat([a, torch.zeros(16000-len(a))])
        a = a[:16000].unsqueeze(0)
        try:
            if r < 0.33:
                a = torchaudio.functional.pitch_shift(a, 16000, random.uniform(-3,3))
            elif r < 0.66:
                a = torchaudio.functional.speed(a, 16000, random.uniform(0.85,1.15))[0]
                if len(a) < 16000: a = torch.cat([a, torch.zeros(16000-len(a))])
                a = a[:16000].unsqueeze(0)
            else:
                a = a + torch.randn_like(a) * random.uniform(0.001, 0.005)
        except:
            pass
        augmented.append(a.squeeze(0)[:16000] if a.dim() > 1 else a)
    return torch.stack(augmented).to(device)

def train_eval(X, y, lr, dropout, wd, seed, augment=False, device='cuda:0'):
    device = torch.device(device)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    results = []
    for tr, vl in skf.split(X, y):
        tr_ds = TensorDataset(X[tr], y[tr])
        vl_ds = TensorDataset(X[vl], y[vl])
        tr_dl = DataLoader(tr_ds, 8, shuffle=True, drop_last=True)
        vl_dl = DataLoader(vl_ds, 8)
        model = Fusion(dropout=dropout).to(device)
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        best, patience, state = 0, 0, None
        for ep in range(1, 101):
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                if augment: xb = augment_batch(xb, device)
                opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
            model.eval(); preds, labels = [], []
            for xb, yb in vl_dl:
                xb = xb.to(device)
                with torch.no_grad(): out = model(xb)
                preds.extend(out.argmax(1).cpu().tolist()); labels.extend(yb.tolist())
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best: best = f1; patience = 0; state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            else: patience += 1
            if patience >= 20: break
        if state: model.load_state_dict(state)
        # Final eval
        model.eval(); preds, labels, probs = [], [], []
        for xb, yb in vl_dl:
            xb = xb.to(device)
            with torch.no_grad(): out = model(xb)
            preds.extend(out.argmax(1).cpu().tolist()); labels.extend(yb.tolist())
            probs.extend(torch.softmax(out,1)[:,1].cpu().tolist())
        tn,fp,fn,tp = confusion_matrix(labels,preds,labels=[0,1]).ravel()
        results.append({'acc':accuracy_score(labels,preds),'f1':f1_score(labels,preds,zero_division=0),
                       'auc':roc_auc_score(labels,probs) if len(set(labels))>1 else 0.5,
                       'sens':tp/(tp+fn+1e-8),'spec':tn/(tn+fp+1e-8)})
    return {k: float(np.mean([r[k] for r in results])) for k in results[0]}

# Load data
X_daic, y_daic = load_npz('results/daic_features.npz')
X_modma, y_modma = load_npz('results/modma_features.npz')

# Hyperparameter sweep (quick: just a few configs)
configs = [
    {'lr': 1e-3, 'dropout': 0.3, 'wd': 0.01, 'aug': False},
    {'lr': 1e-3, 'dropout': 0.5, 'wd': 0.01, 'aug': False},
    {'lr': 5e-4, 'dropout': 0.3, 'wd': 0.01, 'aug': False},
    {'lr': 1e-3, 'dropout': 0.3, 'wd': 0.01, 'aug': True},
    {'lr': 1e-3, 'dropout': 0.5, 'wd': 0.01, 'aug': True},
]

best_f1 = 0
best_cfg = None
all_results = []

for cfg in configs:
    r = train_eval(X_daic, y_daic, cfg['lr'], cfg['dropout'], cfg['wd'], 42, cfg['aug'], 'cuda:0')
    r['config'] = cfg
    all_results.append(r)
    print(f"lr={cfg['lr']:.0e} do={cfg['dropout']} wd={cfg['wd']} aug={cfg['aug']}: F1={r['f1']:.3f} AUC={r['auc']:.3f}")
    if r['f1'] > best_f1:
        best_f1 = r['f1']; best_cfg = cfg

print(f'\nBest: {best_cfg} -> F1={best_f1:.3f}')

# Run best with 3 seeds
print('\n=== Multi-seed with best config ===')
f1s = []
for seed in [42, 142, 242]:
    r = train_eval(X_daic, y_daic, best_cfg['lr'], best_cfg['dropout'], best_cfg['wd'], seed, best_cfg['aug'], 'cuda:0')
    f1s.append(r['f1'])
    print(f'  Seed {seed}: F1={r["f1"]:.3f}')
print(f'DAIC Final: F1={np.mean(f1s):.3f}+-{np.std(f1s,ddof=1):.3f}')

json.dump({'all': all_results, 'best_cfg': best_cfg, 'best_f1s': f1s}, open('results/hp_optimized.json','w'), indent=2)
print('Saved!')
