#!/usr/bin/env python3
"""Standalone multi-classifier comparison using cached wav2vec2 features."""
import json, os, warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

warnings.filterwarnings('ignore')

def cohens_d(g1, g2):
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    g1, g2 = np.asarray(g1), np.asarray(g2)
    s = np.sqrt(((len(g1)-1)*g1.var() + (len(g2)-1)*g2.var()) / (len(g1)+len(g2)-2))
    return (g1.mean() - g2.mean()) / (s + 1e-8)

def run_comparison(X, y, name):
    """Compare LR, SVM, RF, GB, Voting with Cohen's d."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=52)
    
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    k = min(200, X_tr_s.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k)
    X_tr_s = selector.fit_transform(X_tr_s, y_tr)
    X_te_s = selector.transform(X_te_s)
    
    models = {
        "LR": LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", C=1.0),
        "RF": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=52),
        "GB": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05),
        "Voting": VotingClassifier([
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)),
            ("svc", SVC(kernel="rbf", probability=True, class_weight="balanced", C=1.0)),
            ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05)),
        ], voting="soft"),
    }
    
    print(f"\n{'='*90}")
    print(f"  MULTI-CLASSIFIER COMPARISON: {name}")
    print(f"  Cohen's d: effect size between depressed vs non-depressed prediction scores")
    print(f"{'='*90}")
    print(f"  {'Classifier':<12} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Sens':>7} {'Spec':>7} {'Cohen d':>9}  {'Interpretation'}")
    print(f"  {'-'*75}")
    
    results = {}
    for clf_name, clf in models.items():
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_te_s)
        y_proba = clf.predict_proba(X_te_s)[:, 1] if hasattr(clf, "predict_proba") else None
        
        dep = y_proba[y_te == 1] if y_proba is not None else y_pred[y_te == 1].astype(float)
        nondep = y_proba[y_te == 0] if y_proba is not None else y_pred[y_te == 0].astype(float)
        d = cohens_d(dep, nondep)
        
        interp = "negligible" if abs(d) < 0.2 else ("small" if abs(d) < 0.5 else ("medium" if abs(d) < 0.8 else "large"))
        
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        auc = roc_auc_score(y_te, y_proba) if y_proba is not None else np.nan
        sens = recall_score(y_te, y_pred, zero_division=0)
        spec = recall_score(y_te, y_pred, pos_label=0, zero_division=0)
        
        d_str = f"{d:.4f}" if not np.isnan(d) else "N/A"
        print(f"  {clf_name:<12} {acc:7.3f} {f1:7.3f} {auc:7.3f} {sens:7.3f} {spec:7.3f} {d_str:>9}  {interp}")
        
        results[clf_name] = {
            "acc": acc, "f1": f1, "auc": auc, "sens": sens, "spec": spec, "cohens_d": d
        }
    
    print()
    return results

# Load cached features
cache_dir = "/data/disk1/CancerEmotionSystem/results/modma_experiments"
daic = np.load(f"{cache_dir}/daic_w2v2_768d.npz", allow_pickle=True)
modma = np.load(f"{cache_dir}/modma_w2v2_768d.npz", allow_pickle=True)

X_daic, y_daic = daic["X"], daic["y"]
X_modma, y_modma = modma["X"], modma["y"]

print(f"DAIC: {X_daic.shape[0]} subjects, {X_daic.shape[1]} features, {int(y_daic.sum())}/{len(y_daic)} depressed")
print(f"MODMA: {X_modma.shape[0]} subjects, {X_modma.shape[1]} features, {int(y_modma.sum())}/{len(y_modma)} depressed")

daic_results = run_comparison(X_daic, y_daic, "DAIC-WOZ (wav2vec2 768d)")
modma_results = run_comparison(X_modma, y_modma, "MODMA (wav2vec2 768d)")

# Save results
output = {"daic_woz": daic_results, "modma": modma_results}
with open(f"{cache_dir}/classifier_comparison.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"Results saved to {cache_dir}/classifier_comparison.json")
