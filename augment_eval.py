import json, random, numpy as np, torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import torchaudio

def augment_audio(audio, sr=16000):
    """Apply random augmentation to raw audio: pitch, speed, noise"""
    audio = audio.clone()
    r = random.random()
    if r < 0.33:  # pitch shift
        steps = random.uniform(-4, 4)
        audio = torchaudio.functional.pitch_shift(audio.unsqueeze(0), sr, steps).squeeze(0)
    elif r < 0.66:  # time stretch
        rate = random.uniform(0.8, 1.2)
        audio = torchaudio.functional.speed(audio.unsqueeze(0), sr, rate).squeeze(0)
    else:  # gaussian noise
        noise = torch.randn_like(audio) * random.uniform(0.001, 0.01)
        audio = audio + noise
    # Ensure same length as original
    if len(audio) > 480000: audio = audio[:480000]
    elif len(audio) < 480000: audio = torch.cat([audio, torch.zeros(480000 - len(audio))])
    return audio

def load_npz(path):
    d = np.load(path, allow_pickle=True)['data']
    X = np.stack([it['audio'] for it in d])
    y = np.array([float(it['label'] >= 10) for it in d])
    return X, y

for ds_name, npz_path in [('DAIC','results/daic_features.npz'), ('MODMA','results/modma_features.npz')]:
    X, y = load_npz(npz_path)
    print(f'{ds_name}: {X.shape[1]}-dim, {len(y)} subs, dep={y.sum()}')
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    results = []
    for tr, vl in skf.split(X, y):
        # GBDT with tuned parameters for 768-dim
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, 
                                          subsample=0.8, random_state=42)
        clf.fit(X[tr], y[tr])
        preds = clf.predict(X[vl])
        probs = clf.predict_proba(X[vl])[:,1]
        tn,fp,fn,tp = confusion_matrix(y[vl],preds,labels=[0,1]).ravel()
        results.append({'acc':accuracy_score(y[vl],preds), 'f1':f1_score(y[vl],preds,zero_division=0),
                       'auc':roc_auc_score(y[vl],probs), 'sens':tp/(tp+fn+1e-8), 'spec':tn/(tn+fp+1e-8)})
    avg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    std = {k: float(np.std([r[k] for r in results])) for k in results[0]}
    print(f'  GBDT(768d): Acc={avg["acc"]:.3f}+-{std["acc"]:.3f} F1={avg["f1"]:.3f}+-{std["f1"]:.3f} AUC={avg["auc"]:.3f}')
    print(f'  Sens={avg["sens"]:.3f} Spec={avg["spec"]:.3f}')

print('Done - GBDT on 768-dim features')
