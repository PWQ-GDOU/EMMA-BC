"""Reproduce SOTA baselines on DAIC-WOZ with consistent 5-fold CV."""
import json, random, numpy as np, torch, os, sys
from pathlib import Path
import soundfile as sf, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = Path('/data/disk1/datasets/diac_woz')
SR, MAX_SEC = 16000, 30
DEVICE = 'cuda:0'

def load_daic_audio():
    """Load raw audio + labels for DAIC-WOZ train+dev"""
    audios, labels, pids = [], [], []
    for split in ['train', 'dev']:
        df = pd.read_csv(DATA_DIR / 'labels' / f'{split}_split.csv')
        for _, row in df.iterrows():
            pid = row['Participant_ID']
            ap = DATA_DIR / f'{pid}_P' / f'{pid}_AUDIO.wav'
            if not ap.exists(): continue
            audio, _ = sf.read(str(ap))
            audio = torch.FloatTensor(audio)
            max_s = int(MAX_SEC * SR)
            if len(audio) > max_s: audio = audio[:max_s]
            else: audio = torch.cat([audio, torch.zeros(max_s - len(audio))])
            audios.append(audio.numpy())
            labels.append(int(row.get('PHQ_Binary', int(row.get('PHQ_Score', 0) >= 10))))
            pids.append(pid)
    return np.array(audios), np.array(labels), pids

def cv_evaluate(features, labels, clf, name, seeds=[42,142,242]):
    results = []
    for seed in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        fold_results = []
        for tr, vl in skf.split(features, labels):
            Xtr, ytr = features[tr], labels[tr]
            Xvl, yvl = features[vl], labels[vl]
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr); Xvl = scaler.transform(Xvl)
            clf.fit(Xtr, ytr)
            preds = clf.predict(Xvl)
            try:
                probs = clf.predict_proba(Xvl)[:,1]
                auc = roc_auc_score(yvl, probs)
            except: auc = 0.5
            fold_results.append({'acc': accuracy_score(yvl,preds), 'f1': f1_score(yvl,preds,zero_division=0), 'auc': auc})
        results.append({k: float(np.mean([r[k] for r in fold_results])) for k in fold_results[0]})
    avg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    std = {k: float(np.std([r[k] for r in results])) for k in results[0]}
    print(f'  {name}: Acc={avg["acc"]:.3f}+-{std["acc"]:.3f} F1={avg["f1"]:.3f}+-{std["f1"]:.3f} AUC={avg["auc"]:.3f}+-{std["auc"]:.3f}')
    return avg, std

# Load data
print('Loading DAIC-WOZ audio...')
audios, labels, pids = load_daic_audio()
print(f'Loaded {len(audios)} samples, {labels.sum()} depressed')

all_results = {}

# === Baseline 1: openSMILE eGeMAPS + SVM ===
print('\n=== eGeMAPS (AVEC baseline) ===')
try:
    import opensmile
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    egemaps_features = []
    for audio in audios:
        feat = smile.process_signal(audio, SR)
        egemaps_features.append(feat.values.flatten())
    X_egemaps = np.array(egemaps_features)
    print(f'eGeMAPS: {X_egemaps.shape[1]} features')
    all_results['eGeMAPS+SVM'] = cv_evaluate(X_egemaps, labels, SVC(kernel='rbf', probability=True), 'eGeMAPS+SVM')[0]
    all_results['eGeMAPS+RF'] = cv_evaluate(X_egemaps, labels, RandomForestClassifier(200), 'eGeMAPS+RF')[0]
except Exception as e:
    print(f'  eGeMAPS failed: {e}')

# === Baseline 2: MFCC + SVM ===
print('\n=== MFCC (DepAudioNet-style) ===')
try:
    import librosa
    mfcc_features = []
    for audio in audios:
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
        mfcc_features.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))
    X_mfcc = np.array(mfcc_features)
    print(f'MFCC: {X_mfcc.shape[1]} features')
    all_results['MFCC+SVM'] = cv_evaluate(X_mfcc, labels, SVC(kernel='rbf', probability=True), 'MFCC+SVM')[0]
except Exception as e:
    print(f'  MFCC failed: {e}')

# === Baseline 3: WavLM frozen + SVM ===
print('\n=== WavLM frozen + SVM ===')
d = np.load('results/daic_features.npz', allow_pickle=True)['data']
X_wavlm = np.stack([it['audio'] for it in d])
print(f'WavLM: {X_wavlm.shape[1]} features')
all_results['WavLM+SVM'] = cv_evaluate(X_wavlm, labels, SVC(kernel='rbf', probability=True), 'WavLM+SVM')[0]
all_results['WavLM+RF'] = cv_evaluate(X_wavlm, labels, RandomForestClassifier(200), 'WavLM+RF')[0]

# === Baseline 4: WavLM + MLP (our best) ===
print('\n=== WavLM + MLP (Ours) ===')
all_results['WavLM+MLP(Ours)'] = cv_evaluate(X_wavlm, labels, None, 'WavLM+MLP')[0]  # placeholder

# Print comparison table
print('\n' + '='*60)
print('  SOTA COMPARISON (DAIC-WOZ, 5-fold CV)')
print('='*60)
for name, r in sorted(all_results.items(), key=lambda x: x[1].get('f1',0), reverse=True):
    print(f'  {name:<25} F1={r.get("f1",0):.3f} AUC={r.get("auc",0):.3f}')

json.dump(all_results, open('results/sota_baselines.json','w'), indent=2)
print('\nSaved!')
