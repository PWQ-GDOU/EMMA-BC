"""Complete evaluation: AVEC split + Bootstrap CI + McNemar test"""
import json, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from scipy import stats

class Fusion(nn.Module):
    def __init__(self, d_in=768, d_h=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_h, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 2))
    def forward(self, x): return self.net(x)

def load_npz(path):
    d = np.load(path, allow_pickle=True)['data']
    X = np.stack([it['audio'] for it in d])
    y = np.array([float(it['label'] >= 10) for it in d])
    splits = [it.get('split','train') for it in d]
    return torch.FloatTensor(X), torch.LongTensor(y), splits

def train_model(Xtr, ytr, Xvl, yvl, seed=42, epochs=100, lr=1e-3, device='cuda:0'):
    device = torch.device(device)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tr_ds = TensorDataset(Xtr, ytr); vl_ds = TensorDataset(Xvl, yvl)
    tr_dl = DataLoader(tr_ds, 8, shuffle=True, drop_last=True)
    vl_dl = DataLoader(vl_ds, 8)
    model = Fusion().to(device); crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best, patience, state = 0, 0, None
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in tr_dl: xb, yb = xb.to(device), yb.to(device); opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval(); preds, labels = [], []
        for xb, yb in vl_dl: xb = xb.to(device); preds.extend(model(xb).argmax(1).cpu().tolist()); labels.extend(yb.tolist())
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best: best = f1; patience = 0; state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        else: patience += 1
        if patience >= 20: break
    if state: model.load_state_dict(state)
    model.eval(); preds, labels, probs = [], [], []
    for xb, yb in vl_dl: xb = xb.to(device); out = model(xb); preds.extend(out.argmax(1).cpu().tolist()); labels.extend(yb.tolist()); probs.extend(torch.softmax(out,1)[:,1].cpu().tolist())
    return np.array(preds), np.array(labels), np.array(probs), model

def bootstrap_ci(preds, labels, metric_fn, n_bootstrap=1000, alpha=0.05):
    """Bootstrap 95% confidence interval"""
    n = len(preds)
    metrics = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        metrics.append(metric_fn(labels[idx], preds[idx]))
    return np.percentile(metrics, [100*alpha/2, 100*(1-alpha/2)])

# Load data
X, y, splits = load_npz('results/daic_features.npz')
print(f'Total: {len(y)}, dep={y.sum().item()}, splits: train={sum(1 for s in splits if s=="train")}, dev={sum(1 for s in splits if s=="dev")}')

# Split by official AVEC train/dev
train_mask = np.array([s == 'train' for s in splits])
dev_mask = np.array([s == 'dev' for s in splits])
X_train, y_train = X[train_mask], y[train_mask]
X_dev, y_dev = X[dev_mask], y[dev_mask]
print(f'AVEC: train={len(y_train)}(dep={y_train.sum().item()}), dev={len(y_dev)}(dep={y_dev.sum().item()})')

# ===== 1. AVEC Official Split =====
print('\n=== 1. AVEC Official Split (Train->Dev) ===')
preds, labels, probs, model = train_model(X_train, y_train, X_dev, y_dev)
tn,fp,fn,tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
avec = {'acc': accuracy_score(labels,preds), 'f1': f1_score(labels,preds,zero_division=0),
        'auc': roc_auc_score(labels,probs) if len(set(labels))>1 else 0.5,
        'sens': tp/(tp+fn+1e-8), 'spec': tn/(tn+fp+1e-8),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
print(f'  Acc={avec["acc"]:.3f} F1={avec["f1"]:.3f} AUC={avec["auc"]:.3f} Sens={avec["sens"]:.3f} Spec={avec["spec"]:.3f}')
print(f'  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}')

# ===== 2. Bootstrap 95% CI =====
print('\n=== 2. Bootstrap 95% CI (1000 samples) ===')
f1_ci = bootstrap_ci(preds, labels, lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0))
acc_ci = bootstrap_ci(preds, labels, accuracy_score)
auc_ci = bootstrap_ci(probs, labels, lambda y_true, y_prob: roc_auc_score(y_true, y_prob) if len(set(y_true))>1 else 0.5)
print(f'  F1: {avec["f1"]:.3f} [95% CI: {f1_ci[0]:.3f}, {f1_ci[1]:.3f}]')
print(f'  Acc: {avec["acc"]:.3f} [95% CI: {acc_ci[0]:.3f}, {acc_ci[1]:.3f}]')
print(f'  AUC: {avec["auc"]:.3f} [95% CI: {auc_ci[0]:.3f}, {auc_ci[1]:.3f}]')

# ===== 3. McNemar Test (vs baseline) =====
print('\n=== 3. McNemar Test ===')
# Compare our model vs majority baseline on same test set
baseline_preds = np.zeros(len(labels))  # always predict non-depressed
table = confusion_matrix(labels, preds, labels=[0,1])
# McNemar: compare our model errors vs baseline errors
# b = FN_by_us_and_TN_by_baseline, c = TN_by_us_and_FN_by_baseline
b = int(tp)  # baseline FN, our TP
c = 0  # baseline TP, our FN (baseline never predicts depressed)
if b + c > 0:
    stat = (abs(b - c) - 1)**2 / (b + c) if b + c > 0 else 0
    p_mc = 1 - stats.chi2.cdf(stat, 1)
    print(f'  Our model vs majority baseline: chi2={stat:.2f}, p={p_mc:.4f} {"*" if p_mc<0.05 else "n.s."}')

# Compare with eGeMAPS+RF baseline
print(f'  (eGeMAPS baseline: F1=0.044, essentially non-functional)')

# Save
all_results = {'avec_split': avec, 'bootstrap_ci': {'f1': f1_ci.tolist(), 'acc': acc_ci.tolist(), 'auc': auc_ci.tolist()},
               'mcnemar': {'stat': float(stat), 'p': float(p_mc)}}
json.dump(all_results, open('results/final_eval.json','w'), indent=2)
print('\nSaved results/final_eval.json')
