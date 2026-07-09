#!/usr/bin/env python3
"""5-Fold cross-validation clinical evaluation on DAIC-WOZ train+dev (219 participants)."""
import sys, json, os, numpy as np, torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

BASE = Path('/data/disk1/CancerEmotionSystem')
sys.path.insert(0, str(BASE))
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'

from phaseB.multimodal_dataset import DAICWOZDataset, collate_daic
from phaseB.multimodal_model import MultimodalClinicalModel, set_seed
from phaseB.phaseB_train import LabelNormalizer

DEVICE = torch.device('cuda:2')
CKPT = BASE / 'checkpoints/ablation_all/phaseB_best.pt'
N_FOLDS = 5

print('Loading dataset...')
ds = DAICWOZDataset('/data/disk1/datasets/diac_woz', split='train', min_audio_sec=1.0, max_audio_sec=30)
pids = [s['pid'] for s in ds.samples]
labels = []
for s in ds.samples:
    row = ds.phq8_labels.loc[s['pid']]
    labels.append(int(row['PHQ_8Total'] >= 10))
labels = np.array(labels)
print(f'Total: {len(ds)} samples, {len(set(pids))} participants, {labels.sum()} depressed')

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
results = []

for fold, (tr_idx, te_idx) in enumerate(skf.split(range(len(ds)), labels)):
    print(f'\n=== Fold {fold+1}/{N_FOLDS} ===')
    te_ds = Subset(ds, te_idx)
    loader = DataLoader(te_ds, batch_size=2, collate_fn=collate_daic, num_workers=0)
    
    d_video = getattr(ds, 'd_video', None)
    model = MultimodalClinicalModel(d_model=256, n_layers=4, n_heads=8,
                                     freeze_audio_w2v=True, freeze_text_bert=True,
                                     n_tasks=1, d_video=d_video).to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    
    normalizer = LabelNormalizer()
    if 'label_normalizer' in ckpt:
        normalizer.load_state_dict(ckpt['label_normalizer'])
        normalizer.mean = normalizer.mean.to(DEVICE)
        normalizer.std = normalizer.std.to(DEVICE)
    
    model.eval()
    all_preds, all_labels = [], []
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
    labs = torch.cat(all_labels).numpy()
    
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    pb = (preds >= 10).astype(int)
    lb = (labs >= 10).astype(int)
    tn, fp, fn, tp = confusion_matrix(lb, pb, labels=[0,1]).ravel()
    
    fold_r = {
        'fold': fold+1, 'n': len(te_idx),
        'acc': float(accuracy_score(lb, pb)),
        'f1': float(f1_score(lb, pb, zero_division=0)),
        'auc': float(roc_auc_score(lb, preds)) if len(set(lb))>1 else 0.5,
        'sens': float(tp/(tp+fn+1e-8)), 'spec': float(tn/(tn+fp+1e-8)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }
    results.append(fold_r)
    print(f"  Acc={fold_r['acc']:.3f} F1={fold_r['f1']:.3f} AUC={fold_r['auc']:.3f} Sens={fold_r['sens']:.3f} Spec={fold_r['spec']:.3f}")

avgs = {k: float(np.mean([r[k] for r in results])) for k in ['acc','f1','auc','sens','spec']}
stds = {k: float(np.std([r[k] for r in results])) for k in ['acc','f1','auc','sens','spec']}
print(f"\n=== 5-Fold CV Results ===")
print(f"  Acc: {avgs['acc']:.3f} ± {stds['acc']:.3f}")
print(f"  F1:  {avgs['f1']:.3f} ± {stds['f1']:.3f}")
print(f"  AUC: {avgs['auc']:.3f} ± {stds['auc']:.3f}")
print(f"  Sens: {avgs['sens']:.3f} ± {stds['sens']:.3f}")
print(f"  Spec: {avgs['spec']:.3f} ± {stds['spec']:.3f}")

out = {'folds': results, 'avg': avgs, 'std': stds}
with open(BASE/'results/cv_5fold.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to results/cv_5fold.json")
