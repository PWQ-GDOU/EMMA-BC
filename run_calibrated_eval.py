#!/usr/bin/env python3
"""Two-stage: regression -> isotonic calibration -> optimal threshold binary classification."""
import sys, json, os, numpy as np, torch
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, brier_score_loss, precision_recall_curve)
from torch.utils.data import DataLoader

BASE = Path('/data/disk1/CancerEmotionSystem')
sys.path.insert(0, str(BASE))
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'

from phaseB.multimodal_dataset import DAICWOZDataset, collate_daic
from phaseB.multimodal_model import MultimodalClinicalModel
from phaseB.phaseB_train import LabelNormalizer

DEVICE = torch.device('cuda:2')
CKPT = BASE / 'checkpoints/ablation_all/phaseB_best.pt'
N_CAL_FOLDS = 5

def load_model():
    ds = DAICWOZDataset('/data/disk1/datasets/diac_woz', split='train', min_audio_sec=1.0, max_audio_sec=30)
    d_video = getattr(ds, 'd_video', None)
    m = MultimodalClinicalModel(d_model=256, n_layers=4, n_heads=8,
                                 freeze_audio_w2v=True, freeze_text_bert=True,
                                 n_tasks=1, d_video=d_video).to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt['model_state_dict']); m.eval()
    nm = LabelNormalizer()
    if 'label_normalizer' in ckpt:
        nm.load_state_dict(ckpt['label_normalizer'])
        nm.mean = nm.mean.to(DEVICE); nm.std = nm.std.to(DEVICE)
    return m, nm, ds

print('Loading model...')
model, normalizer, dataset = load_model()
print(f'Dataset: {len(dataset)} samples')

# ---- Stage 1: Collect regression predictions ----
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_daic, shuffle=False)
all_preds, all_labels, all_binary = [], [], []
with torch.no_grad():
    for batch in loader:
        audio = batch['audio'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        texts = batch['texts']
        lbs = batch['phq_total']
        if torch.isnan(lbs).any(): continue
        out = model(audio, texts, attention_mask=mask)
        preds = out['phq'].squeeze(-1).cpu()
        denorm = preds * normalizer.std.squeeze().cpu() + normalizer.mean.squeeze().cpu()
        all_preds.append(denorm)
        all_labels.append(lbs)

preds = torch.cat(all_preds).numpy()
labels = torch.cat(all_labels).numpy()
binary = (labels >= 10).astype(int)
print(f'PHQ-8 range: [{preds.min():.1f}, {preds.max():.1f}], true range: [{labels.min():.0f}, {labels.max():.0f}]')
print(f'Depressed: {binary.sum()}/{len(binary)} ({100*binary.sum()/len(binary):.1f}%)')

# ---- Stage 2: Isotonic calibration with 5-fold ----
skf = StratifiedKFold(n_splits=N_CAL_FOLDS, shuffle=True, random_state=42)
calibrated_probs = np.zeros(len(preds))
optimal_thresholds = []

for fold, (cal_idx, val_idx) in enumerate(skf.split(preds.reshape(-1,1), binary)):
    # Fit calibrator
    cal = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    cal.fit(preds[cal_idx], binary[cal_idx])
    cal_probs = cal.predict(preds[val_idx])
    calibrated_probs[val_idx] = cal_probs
    
    # Find optimal threshold on this fold
    best_f1, best_thresh = 0, 10
    for t in np.linspace(2, 15, 100):
        pb = (preds[val_idx] >= t).astype(int)
        f1 = f1_score(binary[val_idx], pb, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_thresh = t
    optimal_thresholds.append(best_thresh)

opt_thresh = np.median(optimal_thresholds)

# ---- Evaluate calibrated ----
# Method 1: Use calibrated probability > 0.5
cal_binary = (calibrated_probs >= 0.5).astype(int)
# Method 2: Use optimized PHQ-8 threshold
thresh_binary = (preds >= opt_thresh).astype(int)
# Method 3: Original naive threshold
naive_binary = (preds >= 10).astype(int)

def metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        'acc': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_prob)) if y_prob is not None and len(set(y_true))>1 else 0.5,
        'sens': float(tp/(tp+fn+1e-8)), 'spec': float(tn/(tn+fp+1e-8)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }

print(f'\n{"="*65}')
print(f'  Two-Stage Calibration Results')
print(f'{"="*65}')
print(f'  Optimal PHQ-8 threshold: {opt_thresh:.1f} (median over {N_CAL_FOLDS} folds)')
print(f'  Individual fold thresholds: {[f"{t:.1f}" for t in optimal_thresholds]}')
print()

m_naive = metrics(binary, naive_binary, preds)
m_thresh = metrics(binary, thresh_binary, preds)
m_cal = metrics(binary, cal_binary, calibrated_probs)

print(f'  {"Method":<25} {"Acc":>6} {"F1":>6} {"AUC":>6} {"Sens":>6} {"Spec":>6} {"TP":>4} {"TN":>4} {"FP":>4} {"FN":>4}')
print(f'  {"-"*70}')
for name, m in [('Naive (PHQ>=10)', m_naive), ('Optimized threshold', m_thresh), ('Isotonic calibration', m_cal)]:
    print(f'  {name:<25} {m["acc"]:6.3f} {m["f1"]:6.3f} {m["auc"]:6.3f} {m["sens"]:6.3f} {m["spec"]:6.3f} {m["tp"]:4d} {m["tn"]:4d} {m["fp"]:4d} {m["fn"]:4d}')

# Brier score
brier_cal = brier_score_loss(binary, calibrated_probs)
print(f'\n  Brier score (calibrated): {brier_cal:.4f}')

out = {
    'optimal_threshold': float(opt_thresh),
    'fold_thresholds': [float(t) for t in optimal_thresholds],
    'naive': m_naive, 'optimized_threshold': m_thresh, 'isotonic_calibrated': m_cal,
    'brier_score': float(brier_cal),
    'n_samples': len(preds), 'n_depressed': int(binary.sum()),
}
with open(BASE/'results/calibrated_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved to results/calibrated_results.json')
