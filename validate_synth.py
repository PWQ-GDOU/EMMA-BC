#!/usr/bin/env python3
"""Layer 3: Quick validation using pre-extracted features."""
import os, sys, glob, json, random
import numpy as np
import torch
import soundfile as sf
from transformers import Wav2Vec2Model
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

print("Loading wav2vec2...")
w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
w2v.eval()

def extract_w2v(audio_path):
    y, sr = sf.read(audio_path)
    if sr != 16000:
        import librosa; y = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
    if len(y) > 16000 * 12:
        y = y[:16000 * 12]
    wav = torch.from_numpy(y).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = w2v(wav)
    return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

# Extract synthetic features
SYNTH = "/data/disk3/EMMA-BC/synthetic"
print("Extracting synthetic features...")
synth_feats = []
for f in sorted(glob.glob(SYNTH + "/synth_*.wav")):
    feat = extract_w2v(f)
    synth_feats.append(feat)
synth_feats = np.stack(synth_feats)
print("Synth:", synth_feats.shape)

# Load real features
REAL = "/data/disk3/EMMA-BC/results/daic_features.npz"
data = np.load(REAL, allow_pickle=True)['data']
real_x, real_y = [], []
for item in data:
    phq = item.get('phq_total', item.get('phq', 0))
    af = item.get('audio_feat', item.get('audio', None))
    if af is not None:
        real_x.append(af)
        real_y.append(1 if phq >= 10 else 0)
real_x = np.array(real_x); real_y = np.array(real_y)
print("Real:", real_x.shape, "Pos:", sum(real_y), "Neg:", len(real_y)-sum(real_y))

# Train: real + synthetic
neg_x = real_x[real_y == 0]
pos_x = real_x[real_y == 1]
X = np.vstack([neg_x, pos_x, synth_feats])
y = np.array([0]*len(neg_x) + [1]*len(pos_x) + [1]*len(synth_feats))
print("Train X:", X.shape, "Y:", y.shape, "Pos:", sum(y))

scaler = StandardScaler()
X = scaler.fit_transform(X)

clf = LogisticRegression(max_iter=2000, class_weight='balanced')
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

f1s, aucs = [], []
for ti, vi in kf.split(X, y):
    clf.fit(X[ti], y[ti])
    p = clf.predict(X[vi])
    pb = clf.predict_proba(X[vi])[:,1]
    f1s.append(f1_score(y[vi], p, zero_division=0))
    aucs.append(roc_auc_score(y[vi], pb) if len(set(y[vi]))>1 else 0.5)

print("=== With Synthetic ===")
print("F1: %.4f +/- %.4f" % (np.mean(f1s), np.std(f1s)))
print("AUC: %.4f +/- %.4f" % (np.mean(aucs), np.std(aucs)))
print("Individual F1:", [round(x,4) for x in f1s])

# Baseline
Xb = np.vstack([neg_x, pos_x])
yb = np.array([0]*len(neg_x) + [1]*len(pos_x))
Xb = scaler.fit_transform(Xb)

f1b, aucb = [], []
for ti, vi in kf.split(Xb, yb):
    clf.fit(Xb[ti], yb[ti])
    p = clf.predict(Xb[vi])
    pb = clf.predict_proba(Xb[vi])[:,1]
    f1b.append(f1_score(yb[vi], p, zero_division=0))
    aucb.append(roc_auc_score(yb[vi], pb) if len(set(yb[vi]))>1 else 0.5)

print("=== Baseline (no synthetic) ===")
print("F1: %.4f +/- %.4f" % (np.mean(f1b), np.std(f1b)))
print("AUC: %.4f +/- %.4f" % (np.mean(aucb), np.std(aucb)))

d_f1 = np.mean(f1s) - np.mean(f1b)
d_auc = np.mean(aucs) - np.mean(aucb)
print("=== Improvement ===")
print("Delta F1: %+.4f" % d_f1)
print("Delta AUC: %+.4f" % d_auc)
print("Result:", "POSITIVE" if d_f1 > 0.01 else "NEGLIGIBLE" if d_f1 > -0.01 else "NEGATIVE")
