#!/usr/bin/env python3
"""
EMMA-BC Route B: Unified Experiment Pipeline
=============================================
1. Extract CLEAN wav2vec2 features (768d mean pooling) from DAIC + MODMA
2. Cross-dataset validation: DAIC <-> MODMA
3. MODMA multi-target regression (PHQ-9 + GAD-7 + PSQI)
4. Summary report

Key design: Both datasets share the SAME 768d feature space for full comparability.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import soundfile as sf
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_regression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, mean_absolute_error, r2_score,
    recall_score, precision_score
)
from scipy.stats import pearsonr
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==========================================================
# CONFIG
# ==========================================================
CONFIG = {
    'device': 'cuda:0',
    'seed': 52,
    'sample_rate': 16000,
    'max_audio_sec': 180,
    'n_cv_folds': 5,
    'w2v2_model': 'facebook/wav2vec2-base',
    'cache_dir': '/data/disk1/models/',
    
    # DAIC paths
    'daic_audio_dir': '/data/disk1/datasets/diac_woz/',
    'daic_split_csv': '/data/disk1/datasets/diac_woz/labels/train_split.csv',
    'daic_dev_split_csv': '/data/disk1/datasets/diac_woz/labels/dev_split.csv',
    
    # MODMA paths
    'modma_audio_dir': '/data/disk1/datasets/modma/extracted/audio_lanzhou_2015/',
    'modma_xlsx': '/data/disk1/datasets/modma/extracted/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx',
    
    # Output
    'output_dir': '/data/disk1/CancerEmotionSystem/results/modma_experiments/',
    
    # Thresholds
    'phq9_depressed': 10,
    'phq8_depressed': 10,
}

# ==========================================================
# UTILITY: Wav2Vec2 Feature Extractor
# ==========================================================
class Wav2Vec2Extractor:
    """Extract 768d features via mean pooling of last_hidden_state."""
    
    def __init__(self, model_name, device='cuda:0', cache_dir=None, local_files_only=False):
        self.device = torch.device(device)
        load_kwargs = {'local_files_only': local_files_only} if local_files_only else {}
        if cache_dir and os.path.exists(cache_dir):
            load_kwargs['cache_dir'] = cache_dir
        self.model = Wav2Vec2Model.from_pretrained(model_name, **load_kwargs).to(self.device)
        self.model.eval()
        self.target_sr = 16000
        print(f'Loaded {model_name} on {device}')
    
    def _process_audio(self, audio, sr=None):
        """Manual preprocessing: resample to 16kHz and normalize."""
        if sr is not None and sr != self.target_sr:
            try:
                import librosa
                audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=self.target_sr)
            except:
                pass
        # Ensure float32 and normalize
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()
        return torch.from_numpy(audio).unsqueeze(0).to(self.device)
    
    def extract(self, audio_path, max_sec=None):
        """Extract 768d features from a single audio file."""
        try:
            audio, sr = sf.read(str(audio_path))
        except Exception as e:
            print(f'  Error reading {audio_path}: {e}')
            return None
        
        if max_sec:
            max_samples = max_sec * self.target_sr
            if len(audio) > max_samples:
                audio = audio[:max_samples]
        
        if len(audio) < self.target_sr:  # min 1 sec
            return None
        
        input_values = self._process_audio(audio, sr=sr)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            # Mean pool over time → 768d
            feat = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        return feat.astype(np.float32)
    
    def extract_concatenated(self, audio_files, max_total_sec=None):
        """Extract features from multiple audio files (concatenated), return mean-pooled 768d."""
        all_audio = []
        for f in audio_files:
            result = self._read_audio(f)
            if result[0] is not None:
                all_audio.append(result)
        
        if not all_audio:
            return None
        
        audio = np.concatenate([a for a, _ in all_audio])
        sr = all_audio[0][1] if all_audio else 16000
        
        if max_total_sec:
            max_samples = max_total_sec * self.target_sr
            if len(audio) > max_samples:
                audio = audio[:max_samples]
        
        if len(audio) < self.target_sr:
            return None
        
        input_values = self._process_audio(audio, sr=sr)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            feat = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        return feat.astype(np.float32)
    
    def _read_audio(self, path):
        try:
            audio, sr = sf.read(str(path))
            return audio, sr
        except:
            return None, None


# ==========================================================
# STEP 0: Load / Prepare Unified Features
# ==========================================================
def load_daic(config, extractor=None):
    """Load DAIC-WOZ data: audio features + PHQ-8 labels."""
    cache_path = os.path.join(config['output_dir'], 'daic_w2v2_768d.npz')
    
    if os.path.exists(cache_path):
        print(f'Loading cached DAIC features from {cache_path}')
        data = np.load(cache_path, allow_pickle=True)
        return data['X'], data['y'], list(data['pids'])
    
    print('Extracting DAIC-WOZ wav2vec2 features (768d)...')
    
    # Load train + dev splits
    labels = {}
    for split_path in [config['daic_split_csv'], config['daic_dev_split_csv']]:
        if os.path.exists(split_path):
            df = pd.read_csv(split_path)
            print(f'Loaded {os.path.basename(split_path)}: {len(df)} rows')
            for _, row in df.iterrows():
                pid = str(int(row['Participant_ID']))
                phq = float(row['PHQ_Score'])
                binary = int(row['PHQ_Binary'])
                labels[pid] = {'phq8': phq, 'binary': binary}
    
    print(f'Total labels loaded: {len(labels)} subjects')
    
    # Audio: /data/disk1/datasets/diac_woz/{pid}_P/{pid}_AUDIO.wav
    audio_dir = Path(config['daic_audio_dir'])
    
    X_list = []
    y_list = []
    pid_list = []
    n_skip = 0
    
    for pid in sorted(labels.keys(), key=int):
        wav_path = audio_dir / f'{pid}_P' / f'{pid}_AUDIO.wav'
        if not wav_path.exists():
            n_skip += 1
            continue
        
        feat = extractor.extract(str(wav_path), max_sec=config['max_audio_sec'])
        if feat is None:
            print(f'  WARNING: Failed to extract for {pid}')
            continue
        
        X_list.append(feat)
        y_list.append(labels[pid]['binary'])
        pid_list.append(pid)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f'Extracted: {len(pid_list)} subjects, X={X.shape}')
    
    # Cache
    os.makedirs(config['output_dir'], exist_ok=True)
    np.savez(cache_path, X=X, y=y, pids=pid_list)
    print(f'Cached to {cache_path}')
    
    return X, y, pid_list


def load_modma(config, extractor=None):
    """Load MODMA data: audio features + clinical labels."""
    cache_path = os.path.join(config['output_dir'], 'modma_w2v2_768d.npz')
    
    if os.path.exists(cache_path):
        print(f'Loading cached MODMA features from {cache_path}')
        data = np.load(cache_path, allow_pickle=True)
        # Load labels dict
        labels_path = os.path.join(config['output_dir'], 'modma_labels.json')
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                labels_dict = json.load(f)
        else:
            labels_dict = None
        return data['X'], data['y'], list(data['pids']), labels_dict
    
    print('Extracting MODMA wav2vec2 features (768d)...')
    
    # Load XLSX
    df = pd.read_excel(config['modma_xlsx'])
    df.columns = [c.strip() for c in df.columns]
    print(f'XLSX columns: {list(df.columns)}')
    
    # Standardize column names
    col_map = {}
    for c in df.columns:
        if 'subject' in c.lower() and 'id' in c.lower():
            col_map[c] = 'subject_id'
        elif 'phq' in c.lower() and '9' in c:
            col_map[c] = 'phq9'
        elif 'gad' in c.lower():
            col_map[c] = 'gad7'
        elif 'psqi' in c.lower():
            col_map[c] = 'psqi'
        elif 'age' in c.lower():
            col_map[c] = 'age'
        elif 'gender' in c.lower():
            col_map[c] = 'gender'
        elif 'educ' in c.lower():
            col_map[c] = 'education'
        elif 'type' in c.lower():
            col_map[c] = 'type'
    
    df = df.rename(columns=col_map)
    
    # Convert subject IDs
    df['subject_id'] = df['subject_id'].astype(str).str.zfill(8)
    
    # Find subject directories
    root = Path(config['modma_audio_dir'])
    subject_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith('02')]
    print(f'Found {len(subject_dirs)} subject directories')
    
    labels_dict = {}
    X_list = []
    y_list = []
    pid_list = []
    
    for subj_dir in tqdm(sorted(subject_dirs), desc='MODMA extract'):
        subj_id = subj_dir.name
        
        subj_row = df[df['subject_id'] == subj_id]
        if len(subj_row) == 0:
            continue
        
        wavs = sorted(subj_dir.glob('*.wav'))
        if not wavs:
            continue
        
        # Concatenate all WAVs and extract
        feat = extractor.extract_concatenated(wavs, max_total_sec=config['max_audio_sec'])
        if feat is None:
            print(f'  Skip {subj_id}: extraction failed')
            continue
        
        X_list.append(feat)
        pid_list.append(subj_id)
        
        # Labels
        row = subj_row.iloc[0]
        label = {}
        for col in ['phq9', 'gad7', 'psqi', 'age', 'gender', 'type']:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    label[col] = float(val) if col not in ['gender', 'type'] else str(val)
        labels_dict[subj_id] = label
        
        # Binary: PHQ-9 >= 10
        binary = 1 if label.get('phq9', 0) >= config['phq9_depressed'] else 0
        y_list.append(binary)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f'Extracted: {len(pid_list)} subjects, X={X.shape}')
    
    # Stats
    phq9s = [l['phq9'] for l in labels_dict.values() if 'phq9' in l]
    if phq9s:
        print(f'PHQ-9: mean={np.mean(phq9s):.2f} std={np.std(phq9s):.2f} '
              f'depressed={sum(1 for s in phq9s if s>=10)}/{len(phq9s)}')
    
    # Cache
    os.makedirs(config['output_dir'], exist_ok=True)
    np.savez(cache_path, X=X, y=y, pids=pid_list)
    with open(os.path.join(config['output_dir'], 'modma_labels.json'), 'w') as f:
        json.dump(labels_dict, f, indent=2)
    print(f'Cached to {cache_path}')
    
    return X, y, pid_list, labels_dict


# ==========================================================
# EVALUATION
# ==========================================================
def evaluate_clf(X_train, y_train, X_test, y_test):
    """VotingClassifier with balanced weights."""
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {k: np.nan for k in ['acc', 'f1', 'auc', 'sens', 'spec']}
    
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    k = min(200, X_train_s.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_s = selector.fit_transform(X_train_s, y_train)
    X_test_s = selector.transform(X_test_s)
    
    clf = VotingClassifier([
        ('lr', LogisticRegression(max_iter=3000, class_weight='balanced', C=0.1)),
        ('svc', SVC(kernel='rbf', probability=True, class_weight='balanced', C=1.0)),
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05)),
    ], voting='soft')
    
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1] if hasattr(clf, 'predict_proba') else None
    
    return {
        'acc': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y_test)) == 2 else np.nan,
        'sens': recall_score(y_test, y_pred, zero_division=0),
        'spec': recall_score(y_test, y_pred, pos_label=0, zero_division=0),
    }


def evaluate_reg(X_train, y_train, X_test, y_test):
    """RidgeCV regression."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    k = min(200, X_train_s.shape[1])
    selector = SelectKBest(f_regression, k=k)
    X_train_s = selector.fit_transform(X_train_s, y_train)
    X_test_s = selector.transform(X_test_s)
    
    model = RidgeCV(alphas=np.logspace(-2, 3, 30))
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    
    r, p = pearsonr(y_test, y_pred)
    
    return {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'pearson_r': r,
        'pearson_p': p,
    }


# ==========================================================
# STEP 1: Within-dataset baselines
# ==========================================================
def within_dataset_cv(X, y, name, n_folds=5, seed=52):
    """5-fold stratified CV on a single dataset."""
    print(f'\n--- {name}: Within-dataset {n_folds}-Fold CV ---')
    
    y_binary = y
    
    # Check class balance
    n_pos = y_binary.sum()
    print(f'  Samples: {len(y_binary)}, Depressed: {int(n_pos)}/{len(y_binary)} '
          f'({100*n_pos/len(y_binary):.1f}%)')
    
    if n_pos < n_folds:
        print(f'  WARNING: Only {int(n_pos)} positive samples, using KFold instead')
        kf = __import__('sklearn.model_selection').model_selection.KFold(
            n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    clf_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y_binary)):
        c = evaluate_clf(X[train_idx], y_binary[train_idx], 
                         X[test_idx], y_binary[test_idx])
        clf_metrics.append(c)
    
    avg = {k: np.nanmean([m[k] for m in clf_metrics]) for k in clf_metrics[0]}
    std = {k: np.nanstd([m[k] for m in clf_metrics]) for k in clf_metrics[0]}
    
    print(f'  Clf: Acc={avg["acc"]:.3f}±{std["acc"]:.3f}  F1={avg["f1"]:.3f}±{std["f1"]:.3f}  '
          f'AUC={avg["auc"]:.3f}±{std["auc"]:.3f}  Sens={avg["sens"]:.3f}  Spec={avg["spec"]:.3f}')
    
    return {'clf_mean': avg, 'clf_std': std}


# ==========================================================
# STEP 2: Cross-dataset evaluation
# ==========================================================
def cross_dataset_eval(X_daic, y_daic, X_modma, y_modma, config):
    """Train on one dataset, test on the other."""
    print(f'\n{"="*60}')
    print('CROSS-DATASET EVALUATION')
    print('='*60)
    print(f'DAIC: {X_daic.shape[0]} subj | MODMA: {X_modma.shape[0]} subj')
    print(f'DAIC feature dim: {X_daic.shape[1]} | MODMA feature dim: {X_modma.shape[1]}')
    
    # Verify same feature dimension
    assert X_daic.shape[1] == X_modma.shape[1], \
        f'Feature dim mismatch: {X_daic.shape[1]} vs {X_modma.shape[1]}'
    
    results = {}
    
    for direction, X_train, y_train, X_test, y_test, train_name, test_name in [
        ('DAIC->MODMA', X_daic, y_daic, X_modma, y_modma, 'DAIC-WOZ', 'MODMA'),
        ('MODMA->DAIC', X_modma, y_modma, X_daic, y_daic, 'MODMA', 'DAIC-WOZ'),
    ]:
        print(f'\n--- {direction}: Train {train_name} → Test {test_name} ---')
        
        clf = evaluate_clf(X_train, y_train, X_test, y_test)
        print(f'  Clf: Acc={clf["acc"]:.3f}  F1={clf["f1"]:.3f}  AUC={clf["auc"]:.3f}  '
              f'Sens={clf["sens"]:.3f}  Spec={clf["spec"]:.3f}')
        results[f'{direction}_clf'] = clf
    
    return results


# ==========================================================
# STEP 3: MODMA multi-target regression
# ==========================================================
def modma_multitarget(X, pid_list, labels_dict, config):
    """Multi-target regression on MODMA: PHQ-9, GAD-7, PSQI."""
    print(f'\n{"="*60}')
    print('MODMA MULTI-TARGET REGRESSION')
    print('='*60)
    
    # Collect target values
    targets = ['phq9', 'gad7', 'psqi']
    y_targets = {}
    valid_pids = []
    
    for pid in pid_list:
        if pid in labels_dict:
            label = labels_dict[pid]
            if all(t in label for t in targets):
                valid_pids.append(pid)
    
    # Map pids to X indices
    pid_to_idx = {p: i for i, p in enumerate(pid_list)}
    valid_indices = [pid_to_idx[p] for p in valid_pids if p in pid_to_idx]
    
    X_valid = X[valid_indices]
    
    for target in targets:
        y_targets[target] = np.array([labels_dict[p][target] for p in valid_pids])
        print(f'{target.upper()}: mean={y_targets[target].mean():.2f} '
              f'std={y_targets[target].std():.2f} '
              f'range=[{y_targets[target].min():.0f}, {y_targets[target].max():.0f}]')
    
    # Binary labels for stratification
    y_binary = (y_targets['phq9'] >= config['phq9_depressed']).astype(int)
    
    results = {}
    
    # Single-task regression
    print('\n--- Single-Task Regression ---')
    n_pos = y_binary.sum()
    if n_pos < config['n_cv_folds']:
        kf = __import__('sklearn.model_selection').model_selection.KFold(
            n_splits=config['n_cv_folds'], shuffle=True, random_state=config['seed'])
    else:
        kf = StratifiedKFold(n_splits=config['n_cv_folds'], shuffle=True, 
                            random_state=config['seed'])
    
    for target in targets:
        y = y_targets[target]
        fold_metrics = []
        
        for train_idx, test_idx in kf.split(X_valid, y_binary):
            r = evaluate_reg(X_valid[train_idx], y[train_idx],
                           X_valid[test_idx], y[test_idx])
            fold_metrics.append(r)
        
        avg = {k: np.nanmean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        std = {k: np.nanstd([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        
        print(f'  {target.upper()}: MAE={avg["mae"]:.2f}±{std["mae"]:.2f}  '
              f'R²={100*avg["r2"]:.1f}±{100*std["r2"]:.1f}%  '
              f'r={avg["pearson_r"]:.3f}±{std["pearson_r"]:.3f}')
        
        results[f'{target}_reg'] = {'mean': avg, 'std': std}
    
    # Multi-task (joint prediction)
    print('\n--- Multi-Task Joint Regression ---')
    y_mt = np.column_stack([y_targets[t] for t in targets])
    mt_metrics = {t: [] for t in targets}
    
    for train_idx, test_idx in kf.split(X_valid, y_binary):
        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_valid[train_idx])
        X_test_s = scaler.transform(X_valid[test_idx])
        
        # Feature selection (shared)
        k = min(200, X_train_s.shape[1])
        # Use average of all targets for feature selection
        y_avg = y_mt[train_idx].mean(axis=1)
        selector = SelectKBest(f_regression, k=k)
        X_train_s = selector.fit_transform(X_train_s, y_avg)
        X_test_s = selector.transform(X_test_s)
        
        for i, target in enumerate(targets):
            y_i = y_mt[train_idx, i]
            
            # Z-score normalize target for RidgeCV
            y_mean_i = y_i.mean()
            y_std_i = y_i.std()
            y_norm = (y_i - y_mean_i) / y_std_i
            
            model = RidgeCV(alphas=np.logspace(-2, 3, 30))
            model.fit(X_train_s, y_norm)
            
            y_pred_norm = model.predict(X_test_s)
            y_pred = y_pred_norm * y_std_i + y_mean_i
            
            r, p = pearsonr(y_mt[test_idx, i], y_pred)
            mt_metrics[target].append({
                'mae': mean_absolute_error(y_mt[test_idx, i], y_pred),
                'r2': r2_score(y_mt[test_idx, i], y_pred),
                'pearson_r': r,
            })
    
    for target in targets:
        avg = {k: np.nanmean([m[k] for m in mt_metrics[target]]) for k in mt_metrics[target][0]}
        std = {k: np.nanstd([m[k] for m in mt_metrics[target]]) for k in mt_metrics[target][0]}
        print(f'  {target.upper()}: MAE={avg["mae"]:.2f}±{std["mae"]:.2f}  '
              f'R²={100*avg["r2"]:.1f}±{100*std["r2"]:.1f}%  '
              f'r={avg["pearson_r"]:.3f}±{std["pearson_r"]:.3f}')
        results[f'{target}_multitask'] = {'mean': avg, 'std': std}
    
    return results


# ==========================================================
# SUMMARY TABLE
# ==========================================================
def print_summary(daic_cv, modma_cv, cross_results, modma_reg):
    """Print comprehensive results table."""
    print(f'\n{"="*80}')
    print('FINAL RESULTS SUMMARY')
    print('='*80)
    
    def v(d, k, fmt='.3f'):
        val = d.get(k, np.nan)
        if np.isnan(val):
            return '  N/A  '
        return f'{val:{fmt}}'
    
    print(f'\n{"Experiment":<30} {"Acc":>7} {"F1":>7} {"AUC":>7} {"Sens":>7} {"Spec":>7}')
    print('-' * 65)
    
    # Within-dataset
    if daic_cv:
        m = daic_cv['clf_mean']
        print(f'{"DAIC-WOZ within 5-CV":<30} {v(m,"acc"):>7} {v(m,"f1"):>7} {v(m,"auc"):>7} {v(m,"sens"):>7} {v(m,"spec"):>7}')
    
    if modma_cv:
        m = modma_cv['clf_mean']
        print(f'{"MODMA within 5-CV":<30} {v(m,"acc"):>7} {v(m,"f1"):>7} {v(m,"auc"):>7} {v(m,"sens"):>7} {v(m,"spec"):>7}')
    
    # Cross-dataset
    if cross_results:
        for direction in ['DAIC->MODMA', 'MODMA->DAIC']:
            key = f'{direction}_clf'
            if key in cross_results:
                m = cross_results[key]
                print(f'{direction:<30} {v(m,"acc"):>7} {v(m,"f1"):>7} {v(m,"auc"):>7} {v(m,"sens"):>7} {v(m,"spec"):>7}')
    
    # Regression
    if modma_reg:
        print(f'\n{"Target":<15} {"MAE":>7} {"R²%":>7} {"Pearson r":>9}')
        print('-' * 40)
        for target in ['phq9', 'gad7', 'psqi']:
            key = f'{target}_reg'
            if key in modma_reg:
                m = modma_reg[key]['mean']
                print(f'{target.upper():<15} {v(m,"mae",".2f"):>7} {v(m,"r2",".1f"):>7} {v(m,"pearson_r"):>9}')


# ==========================================================
# MAIN
# ==========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', default='all',
                       choices=['all', 'extract', 'cross', 'modma_reg', 'summary'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output_dir', default=CONFIG['output_dir'])
    parser.add_argument('--skip_extract', action='store_true',
                       help='Use cached features if available')
    args = parser.parse_args()
    
    config = CONFIG.copy()
    config['device'] = args.device
    config['output_dir'] = args.output_dir
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize extractor (try local first, fallback to mirror)
    try:
        extractor = Wav2Vec2Extractor(config['w2v2_model'], device=config['device'], 
                                       cache_dir=config['cache_dir'], local_files_only=True)
        print('Loaded model from local cache')
    except Exception as e:
        print(f'Local load failed ({e}), trying with HF mirror...')
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        extractor = Wav2Vec2Extractor(config['w2v2_model'], device=config['device'], 
                                       cache_dir=config['cache_dir'], local_files_only=False)
    
    daic_cv = None
    modma_cv = None
    cross_results = None
    modma_reg = None
    
    # ======== STEP 1: Extract features ========
    if args.step in ('all', 'extract'):
        print('\n' + '='*60)
        print('EXTRACTING FEATURES')
        print('='*60)
        
        X_daic, y_daic, daic_pids = load_daic(config, extractor)
        X_modma, y_modma, modma_pids, labels_dict = load_modma(config, extractor)
    
    # ======== STEP 2: Cross-dataset ========
    if args.step in ('all', 'cross'):
        # Reload cached
        daic_cache = os.path.join(config['output_dir'], 'daic_w2v2_768d.npz')
        modma_cache = os.path.join(config['output_dir'], 'modma_w2v2_768d.npz')
        
        d = np.load(daic_cache, allow_pickle=True)
        X_daic, y_daic, daic_pids = d['X'], d['y'], list(d['pids'])
        
        d = np.load(modma_cache, allow_pickle=True)
        X_modma, y_modma, modma_pids = d['X'], d['y'], list(d['pids'])
        
        # Within-dataset
        daic_cv = within_dataset_cv(X_daic, y_daic, 'DAIC-WOZ', n_folds=config['n_cv_folds'])
        modma_cv = within_dataset_cv(X_modma, y_modma, 'MODMA', n_folds=config['n_cv_folds'])
        
        # Cross-dataset
        cross_results = cross_dataset_eval(X_daic, y_daic, X_modma, y_modma, config)
    
    # ======== STEP 3: MODMA multi-target ========
    if args.step in ('all', 'modma_reg'):
        modma_cache = os.path.join(config['output_dir'], 'modma_w2v2_768d.npz')
        labels_path = os.path.join(config['output_dir'], 'modma_labels.json')
        
        d = np.load(modma_cache, allow_pickle=True)
        X_modma, y_modma, modma_pids = d['X'], d['y'], list(d['pids'])
        
        with open(labels_path) as f:
            labels_dict = json.load(f)
        
        modma_reg = modma_multitarget(X_modma, modma_pids, labels_dict, config)
    
    # ======== SUMMARY ========
    print_summary(daic_cv, modma_cv, cross_results, modma_reg)
    
    # Save all results
    all_results = {
        'daic_cv': daic_cv,
        'modma_cv': modma_cv,
        'cross_dataset': cross_results,
        'modma_multitarget': modma_reg,
    }
    
    results_path = os.path.join(config['output_dir'], 'all_results.json')
    # Convert to serializable
    def clean(d):
        if isinstance(d, dict):
            return {k: clean(v) for k, v in d.items()}
        elif isinstance(d, (np.float64, np.float32)):
            return float(d)
        elif isinstance(d, (np.int64, np.int32)):
            return int(d)
        elif isinstance(d, np.ndarray):
            return d.tolist()
        elif isinstance(d, list):
            return [clean(x) for x in d]
        return d
    
    with open(results_path, 'w') as f:
        json.dump(clean(all_results), f, indent=2)
    print(f'\nFull results saved to: {results_path}')
    
    print('\nDone!')


if __name__ == '__main__':
    main()
