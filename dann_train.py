#!/usr/bin/env python3
"""DANN: Domain-Adversarial training to align synthetic and real features."""
import os, sys, glob, json
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from transformers import Wav2Vec2Model
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ---- 1. Extract features ----
print("Loading wav2vec2...")
w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
w2v.eval()

def extract_w2v(audio_path):
    y, sr = sf.read(audio_path)
    if sr != 16000:
        import librosa; y = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
    if len(y) > 16000 * 12: y = y[:16000 * 12]
    wav = torch.from_numpy(y).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = w2v(wav)
    return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

SYNTH = "/data/disk3/EMMA-BC/synthetic"
print("Extracting synthetic...")
synth_x = np.stack([extract_w2v(f) for f in sorted(glob.glob(SYNTH + "/synth_*.wav"))])
print("Synth:", synth_x.shape)

REAL = "/data/disk3/EMMA-BC/results/daic_features.npz"
data = np.load(REAL, allow_pickle=True)['data']
real_x, real_phq = [], []
for item in data:
    af = item.get('audio_feat', item.get('audio', None))
    phq = item.get('phq_total', item.get('phq', 0))
    if af is not None:
        real_x.append(af)
        real_phq.append(phq)
real_x = np.array(real_x); real_phq = np.array(real_phq)
real_y = (real_phq >= 10).astype(int)
print("Real:", real_x.shape, "Pos:", sum(real_y), "Neg:", len(real_y)-sum(real_y))

# If no real positives, just use regression
if real_y.sum() == 0:
    real_y = np.where(real_phq >= np.median(real_phq), 1, 0)
    print("Binarized by median:", sum(real_y), "positives")

# ---- 2. DANN Model ----
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DANN(nn.Module):
    def __init__(self, d_in=768, d_hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(d_hidden, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(), nn.Dropout(0.3),
        )
        self.task_head = nn.Linear(d_hidden, 1)     # depression (regression)
        self.domain_head = nn.Sequential(
            nn.Linear(d_hidden, 128), nn.ReLU(),
            nn.Linear(128, 1),                       # real(0) vs synth(1)
        )
    
    def forward(self, x, alpha=1.0):
        f = self.shared(x)
        task_out = self.task_head(f).squeeze(-1)
        f_rev = GradientReversal.apply(f, alpha)
        domain_out = self.domain_head(f_rev).squeeze(-1)
        return task_out, domain_out

# ---- 3. Training ----
print("\n=== DANN Cross-Validation ===")
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Build full dataset
X_all = np.vstack([real_x, synth_x])
y_all = np.concatenate([real_y, np.ones(len(synth_x))])  # task labels
d_all = np.concatenate([np.zeros(len(real_x)), np.ones(len(synth_x))])  # domain: 0=real,1=synth
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)

def train_fold(X_tr, y_tr, d_tr, X_val, y_val, d_val, fold):
    model = DANN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    task_crit = nn.MSELoss()
    domain_crit = nn.BCEWithLogitsLoss()
    
    Xt = torch.FloatTensor(X_tr).to(DEVICE)
    yt = torch.FloatTensor(y_tr).to(DEVICE)
    dt = torch.FloatTensor(d_tr).to(DEVICE)
    Xv = torch.FloatTensor(X_val).to(DEVICE)
    yv = torch.FloatTensor(y_val).to(DEVICE)
    
    for ep in range(200):
        model.train()
        alpha = min(1.0, ep / 50.0)  # gradually increase GRL strength
        
        t_pred, d_pred = model(Xt, alpha)
        t_loss = task_crit(t_pred, yt)
        d_loss = domain_crit(d_pred, dt)
        loss = t_loss + 0.5 * d_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    model.eval()
    with torch.no_grad():
        t_val, _ = model(Xv, alpha=0)
        t_val = t_val.cpu().numpy()
    
    # Binary classification at threshold 0.5 (normalized labels)
    yv_np = y_val
    t_bin = (t_val >= 0.5).astype(int)
    f1 = f1_score(yv_np, t_bin, zero_division=0)
    if len(set(yv_np)) > 1:
        auc = roc_auc_score(yv_np, t_val)
    else:
        auc = 0.5
    return f1, auc

results_dann = []
results_base = []

for fold, (ti, vi) in enumerate(kf.split(X_all, y_all)):
    print(f"\nFold {fold+1}/{n_folds}...", end=" ", flush=True)
    
    # DANN
    f1_d, auc_d = train_fold(X_all[ti], y_all[ti], d_all[ti], X_all[vi], y_all[vi], d_all[vi], fold)
    results_dann.append((f1_d, auc_d))
    
    # Baseline (same model, no domain loss, no synth)
    real_idx = np.where(d_all[ti] == 0)[0]
    val_idx = np.where(d_all[vi] == 0)[0]  # validate only on real
    if len(real_idx) > 0 and len(val_idx) > 0:
        f1_b, auc_b = train_fold(
            X_all[ti][real_idx], y_all[ti][real_idx], d_all[ti][real_idx],
            X_all[vi][val_idx], y_all[vi][val_idx], d_all[vi][val_idx], fold
        )
        results_base.append((f1_b, auc_b))
    print(f"DANN F1={f1_d:.3f} AUC={auc_d:.3f}")

print("\n=== Results ===")
if results_dann:
    f1 = [r[0] for r in results_dann]
    auc = [r[1] for r in results_dann]
    print(f"DANN:    F1={np.mean(f1):.4f}+/-{np.std(f1):.4f}  AUC={np.mean(auc):.4f}+/-{np.std(auc):.4f}")
    print(f"         Individual F1: {[round(x,4) for x in f1]}")
if results_base:
    f1b = [r[0] for r in results_base]
    aucb = [r[1] for r in results_base]
    print(f"Baseline: F1={np.mean(f1b):.4f}+/-{np.std(f1b):.4f}  AUC={np.mean(aucb):.4f}+/-{np.std(aucb):.4f}")
