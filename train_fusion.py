#!/usr/bin/env python3
"""Cross-Attention Fusion on pre-extracted features. MODMA pretrain -> DAIC finetune."""
import json, random, numpy as np, torch, torch.nn as nn, argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

class CrossAttnFusion(nn.Module):
    def __init__(self, d_audio=768, d_text=768, d_video=49, d_model=256, n_heads=4, dropout=0.3, n_classes=1):
        super().__init__()
        self.audio_proj = nn.Sequential(nn.Linear(d_audio, d_model), nn.GELU(), nn.Dropout(dropout))
        self.text_proj = nn.Sequential(nn.Linear(d_text, d_model), nn.GELU(), nn.Dropout(dropout))
        self.video_proj = nn.Sequential(nn.Linear(d_video, d_model), nn.GELU(), nn.Dropout(dropout))
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, audio, text, video, mode='full'):
        a = self.audio_proj(audio).unsqueeze(1)
        t = self.text_proj(text).unsqueeze(1)
        v = self.video_proj(video).unsqueeze(1) if mode != 'audio_text' else torch.zeros_like(a)
        if mode == 'audio_only': t, v = t*0, v*0
        elif mode == 'text_only': a, v = a*0, v*0
        elif mode == 'video_only': a, t = a*0, t*0
        elif mode == 'audio_video': t = t*0
        elif mode == 'video_text': a = a*0
        x = torch.cat([a, t, v], dim=1)
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ffn(x))
        return self.classifier(x.mean(dim=1)).squeeze(-1)

def load_data(npz_path):
    d = np.load(npz_path, allow_pickle=True)['data']
    X_a, X_t, X_v, y = [], [], [], []
    for item in d:
        X_a.append(item.get("audio", np.zeros(768)))
        X_t.append(item.get("text", np.zeros(768)))
        X_v.append(item.get("video", np.zeros(49)))
        y.append(float(item['label'] >= 10))
    return torch.FloatTensor(np.stack(X_a)), torch.FloatTensor(np.stack(X_t)), torch.FloatTensor(np.stack(X_v)), torch.FloatTensor(y)

def evaluate(model, loader, mode, device):
    model.eval(); preds, labels, probs = [], [], []
    for a, t, v, y in loader:
        a, t, v, y = a.to(device), t.to(device), v.to(device), y.to(device)
        with torch.no_grad():
            out = model(a, t, v, mode)
        pb = out.argmax(1); lb = y.long()
        preds.extend(pb.cpu().tolist()); labels.extend(lb.cpu().tolist())
        probs.extend(torch.softmax(out,1)[:,1].cpu().tolist())
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    return {'acc': accuracy_score(labels,preds), 'f1': f1_score(labels,preds,zero_division=0),
            'auc': roc_auc_score(labels,probs) if len(set(labels))>1 else 0.5,
            'sens': tp/(tp+fn+1e-8), 'spec': tn/(tn+fp+1e-8)}

def train_one(model, tr_loader, te_loader, mode, device, epochs=100, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()  # will compute weights from data
    best, patience, state = 0, 0, None
    for ep in range(1, epochs+1):
        model.train()
        for a, t, v, y in tr_loader:
            a, t, v, y = a.to(device), t.to(device), v.to(device), y.to(device)
            opt.zero_grad(); crit(model(a, t, v, mode), y.long()).backward(); opt.step()
        vm = evaluate(model, te_loader, mode, device)
        if vm['f1'] > best: best = vm['f1']; patience = 0; state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        else: patience += 1
        if patience >= 20: break
    if state: model.load_state_dict(state)
    return evaluate(model, te_loader, mode, device)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='daic', choices=['modma','daic','transfer'])
    p.add_argument('--modality', default='full', choices=['full','audio_only','text_only','video_only','audio_video','video_text','audio_text'])
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    device = torch.device(args.device)
    
    if args.mode == 'modma':
        Xa, Xt, Xv, y = load_data('results/modma_features.npz')
        Xt = Xt * 0  # MODMA has no transcripts
        print(f'MODMA: {len(y)} subjects, depressed={y.sum().item()}')
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        results = []
        for tr, vl in skf.split(Xa, (y>=10).long()):
            tr_ds = TensorDataset(Xa[tr], Xt[tr], Xv[tr], y[tr])
            vl_ds = TensorDataset(Xa[vl], Xt[vl], Xv[vl], y[vl])
            tr_dl = DataLoader(tr_ds, 8, shuffle=True, drop_last=True)
            vl_dl = DataLoader(vl_ds, 8)
            model = CrossAttnFusion().to(device)
            r = train_one(model, tr_dl, vl_dl, args.modality, device)
            results.append(r)
            print(f'  Fold: f1={r["f1"]:.3f} auc={r["auc"]:.3f}')
        avg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
        std = {k: float(np.std([r[k] for r in results])) for k in results[0]}
        print(f'MODMA {args.modality}: Acc={avg["acc"]:.3f}+-{std["acc"]:.3f} F1={avg["f1"]:.3f}+-{std["f1"]:.3f} AUC={avg["auc"]:.3f}')
        json.dump({'avg':avg,'std':std,'folds':[{k:float(v) for k,v in r.items()} for r in results]}, open(f'results/modma_{args.modality}.json','w'))
    
    elif args.mode == 'daic':
        Xa, Xt, Xv, y = load_data('results/daic_features.npz')
        print(f'DAIC: {len(y)} subjects, depressed={y.sum().item()}')
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        results = []
        for tr, vl in skf.split(Xa, (y>=10).long()):
            tr_ds = TensorDataset(Xa[tr], Xt[tr], Xv[tr], y[tr])
            vl_ds = TensorDataset(Xa[vl], Xt[vl], Xv[vl], y[vl])
            tr_dl = DataLoader(tr_ds, 8, shuffle=True, drop_last=True)
            vl_dl = DataLoader(vl_ds, 8)
            model = CrossAttnFusion().to(device)
            r = train_one(model, tr_dl, vl_dl, args.modality, device)
            results.append(r)
            print(f'  Fold: f1={r["f1"]:.3f} auc={r["auc"]:.3f} sens={r["sens"]:.3f} spec={r["spec"]:.3f}')
        avg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
        std = {k: float(np.std([r[k] for r in results])) for k in results[0]}
        print(f'DAIC {args.modality}: Acc={avg["acc"]:.3f}+-{std["acc"]:.3f} F1={avg["f1"]:.3f}+-{std["f1"]:.3f} AUC={avg["auc"]:.3f}')
        json.dump({'avg':avg,'std':std}, open(f'results/daic_{args.modality}.json','w'))

if __name__ == '__main__':
    main()
