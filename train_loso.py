#!/usr/bin/env python3
import os, sys, json, random, argparse, warnings
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from pathlib import Path
import soundfile as sf, pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')
BASE = Path('/data/disk1/CancerEmotionSystem')
sys.path.insert(0, str(BASE))
from src.models.wavlm_adapter import DepressionClassifier
BATCH, EPOCHS, LR = 1, 30, 2e-4
SR, MAX_SEC = 16000, 30
DEVICE = torch.device('cuda:2')
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'

def load_daic_data(split='train'):
    data_dir = Path('/data/disk1/datasets/diac_woz')
    csv_path = data_dir / 'labels' / f'{split}_split.csv'
    if not csv_path.exists(): return []
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        pid = row['Participant_ID']
        ap = data_dir / f'{pid}_P' / f'{pid}_AUDIO.wav'
        if not ap.exists(): continue
        try:
            audio_raw, _ = sf.read(str(ap))
            audio = torch.FloatTensor(audio_raw)
        except: continue
        max_s = int(MAX_SEC * SR)
        if len(audio) > max_s: audio = audio[:max_s]
        else: audio = torch.cat([audio, torch.zeros(max_s - len(audio))])
        transcript = ''
        tp = data_dir / f'{pid}_P' / f'{pid}_Transcript.csv'
        if tp.exists():
            try:
                tdf = pd.read_csv(tp)
                transcript = ' '.join(tdf['Text'].dropna().tolist())
            except: pass
        label = int(row.get('PHQ_Binary', int(row.get('PHQ_Score', 0) >= 10)))
        samples.append({'pid': pid, 'audio': audio, 'transcript': transcript, 'label': label})
    return samples

def collate_fn(batch):
    audios = [b['audio'] for b in batch]
    max_l = max(a.shape[0] for a in audios)
    padded = torch.zeros(len(audios), max_l)
    mask = torch.zeros(len(audios), max_l)
    for i, a in enumerate(audios): padded[i,:a.shape[0]] = a; mask[i,:a.shape[0]] = 1
    texts = [b['transcript'] for b in batch]
    labels = torch.LongTensor([b['label'] for b in batch])
    return padded, mask, texts, None, labels

def train_epoch(model, loader, opt, crit):
    model.train(); tl, tc, tt = 0, 0, 0
    for audio, amask, texts, _, labels in loader:
        audio, amask, labels = audio.to(DEVICE), amask.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        logits = model(audio, texts, None, amask, None)
        loss = crit(logits, labels); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item(); tc += (logits.argmax(1)==labels).sum().item(); tt += len(labels)
    return tl/len(loader), tc/tt

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds, labels, probs = [], [], []
    for audio, amask, texts, _, lbls in loader:
        audio, amask, lbls = audio.to(DEVICE), amask.to(DEVICE), lbls.to(DEVICE)
        logits = model(audio, texts, None, amask, None)
        preds.extend(logits.argmax(1).cpu().tolist()); labels.extend(lbls.cpu().tolist())
        probs.extend(torch.softmax(logits,1)[:,1].cpu().tolist())
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    return {'acc': accuracy_score(labels,preds), 'f1': f1_score(labels,preds,zero_division=0),
            'auc': roc_auc_score(labels,probs) if len(set(labels))>1 else 0.5,
            'sens': tp/(tp+fn+1e-8), 'spec': tn/(tn+fp+1e-8)}

def run_experiment(train_data, test_data, seed, name=''):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tr_dl = DataLoader(train_data, BATCH, shuffle=True, collate_fn=collate_fn, drop_last=True)
    te_dl = DataLoader(test_data, BATCH, collate_fn=collate_fn)
    model = DepressionClassifier().to(DEVICE)
    print(f'  {name}: trainable={model.trainable_params():,}')
    lbs = [s['label'] for s in train_data]; pos = sum(lbs); neg = len(lbs) - pos
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, neg/max(pos,1)], device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    best_f1, patience, state = 0, 0, None
    for ep in range(1, EPOCHS+1):
        tl, ta = train_epoch(model, tr_dl, opt, crit)
        vm = evaluate(model, te_dl)
        if vm['f1'] > best_f1: best_f1 = vm['f1']; patience = 0; state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        else: patience += 1
        if ep % 10 == 0: print(f'    Ep{ep}: loss={tl:.3f} f1={vm["f1"]:.3f} auc={vm["auc"]:.3f}')
        if patience >= 15: break
    if state: model.load_state_dict(state)
    return evaluate(model, te_dl)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--output', default='results/wavlm_results.json')
    args = parser.parse_args()
    global DEVICE; DEVICE = torch.device(args.device)
    print('Loading DAIC-WOZ...')
    train = load_daic_data('train')
    dev = load_daic_data('dev')
    print(f'Train: {len(train)}, Dev: {len(dev)}')
    print(f'Depressed: train={sum(s["label"] for s in train)}/{len(train)}, dev={sum(s["label"] for s in dev)}/{len(dev)}')
    print("
=== AVEC Split ===")
=== AVEC Split (Train->Dev) ===") #'
=== AVEC Split (Train->Dev) ===')
    r = run_experiment(train, dev, 42, 'AVEC')
    print(f'AVEC: Acc={r["acc"]:.3f} F1={r["f1"]:.3f} AUC={r["auc"]:.3f} Sens={r["sens"]:.3f} Spec={r["spec"]:.3f}')
    with open(args.output, 'w') as f: json.dump({'avec': {k:float(v) if isinstance(v,(np.floating,float)) else v for k,v in r.items()}}, f, indent=2)
    print(f'Saved to {args.output}')

if __name__ == '__main__':
    main()
