#!/usr/bin/env python3
"""Compare 3 approaches to fix sensitivity on DAIC+MODMA."""
import os, sys
sys.path.insert(0, "/data/disk3/EMMA-BC")
sys.path.insert(0, "/data/disk3/EMMA-BC/phaseB")

import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from multimodal_dataset import DAICWOZDataset, MODMADataset, CombinedMultimodalDataset, collate_combined
from multimodal_model import MultimodalClinicalModel, set_seed

DEVICE = torch.device("cuda:0")
set_seed(42)

# Load data once
daic = DAICWOZDataset("/data/disk1/datasets/diac_woz", "train", False, 1.0, 5)
modma = MODMADataset("/data/disk1/datasets/modma/extracted",
    "/data/disk1/datasets/modma/extracted/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx", 120)
full = CombinedMultimodalDataset(daic, modma)
train_ds, val_ds = full.split_by_subject(0.15, 42)
print("Train:", len(train_ds), "Val:", len(val_ds))

tl = DataLoader(train_ds, 2, True, collate_fn=collate_combined, num_workers=0, pin_memory=True, drop_last=True)
vl = DataLoader(val_ds, 2, False, collate_fn=collate_combined, num_workers=0, pin_memory=True)

# Filter bad samples from both datasets
def filter_ds(ds):
    good = []
    for i in range(len(ds)):
        try:
            _ = ds[i]; good.append(i)
        except: pass
    from torch.utils.data import Subset
    return Subset(ds, good), len(good)

train_ds, n_good_t = filter_ds(train_ds)
val_ds, n_good_v = filter_ds(val_ds)
print("Good: train=%d/%d val=%d/%d" % (n_good_t, 98, n_good_v, 16))
tl = DataLoader(train_ds, 2, True, collate_fn=collate_combined, num_workers=0, pin_memory=True, drop_last=True)
vl = DataLoader(val_ds, 2, False, collate_fn=collate_combined, num_workers=0, pin_memory=True)

def eval_model(model):
    model.eval()
    preds, lbls, probs = [], [], []
    with torch.no_grad():
        for b in vl:
            a = b["audio"].to(DEVICE); m = b["attention_mask"].to(DEVICE)
            l = b["phq_total"]
            out = model(a, [""], attention_mask=m, has_text=torch.zeros(a.shape[0], dtype=torch.bool).to(DEVICE))
            p = out["phq"].squeeze(-1).cpu()
            # denormalize
            p = p * 7.0 + 8.0
            preds.append(p); lbls.append(l)
    preds = torch.cat(preds); lbls = torch.cat(lbls)
    mae = (preds - lbls).abs().mean().item()
    pb = (preds >= 10).int(); lb = (lbls >= 10).int()
    tp = ((pb==1)&(lb==1)).sum().item(); tn = ((pb==0)&(lb==0)).sum().item()
    fp = ((pb==1)&(lb==0)).sum().item(); fn = ((pb==0)&(lb==1)).sum().item()
    sens = tp/(tp+fn+1e-8); spec = tn/(tn+fp+1e-8)
    return mae, sens, spec, tp, tn, fp, fn

def train_one(model, opt, loss_fn, epochs=25):
    best_sens = -1
    for ep in range(1, epochs+1):
        model.train()
        for b in tl:
            a = b["audio"].to(DEVICE); m = b["attention_mask"].to(DEVICE)
            l = b["phq_total"].to(DEVICE)
            out = model(a, [""], attention_mask=m, has_text=torch.zeros(a.shape[0], dtype=torch.bool).to(DEVICE))
            pred = out["phq"].squeeze(-1)
            # normalize labels
            l_norm = (l - 8.0) / 7.0
            loss = loss_fn(pred, l_norm, l)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ep % 5 == 0:
            mae, sens, spec, tp, tn, fp, fn = eval_model(model)
            print("  E%02d MAE=%.2f Sens=%.3f Spec=%.3f" % (ep, mae, sens, spec))
            if sens > best_sens: best_sens = sens
    return eval_model(model) + (best_sens,)

# ---- Experiment 1: Weighted MSE (3x for depressed) ----
print("\n=== Exp1: Weighted MSE (3x for PHQ>=10) ===")
model1 = MultimodalClinicalModel(256, 4, 8, True, True, 1).to(DEVICE)
model1.freeze_backbones_for_training()
opt1 = torch.optim.AdamW([p for p in model1.parameters() if p.requires_grad], 5e-4)

class WeightedMSE(nn.Module):
    def forward(self, pred, target_norm, target_raw):
        mse = (pred - target_norm) ** 2
        weight = torch.where(target_raw >= 10, 3.0, 1.0)
        return (weight * mse).mean()

r1 = train_one(model1, opt1, WeightedMSE())
print("Exp1 Best: MAE=%.3f Sens=%.3f Spec=%.3f" % (r1[0], r1[1], r1[2]))

# ---- Experiment 2: Binary Classification ----
print("\n=== Exp2: Binary Classification (>=10) ===")

class BinaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MultimodalClinicalModel(256, 4, 8, True, True, 1)
        self.backbone.freeze_backbones_for_training()
        # Replace regression head with binary
        in_dim = list(self.backbone.fusion_head.heads["phq"].children())[-1].in_features
        self.binary_head = nn.Linear(in_dim, 1)
        # Redirect: we need the fusion features, not PHQ score
    def forward(self, waveform, texts, attention_mask=None, has_text=None):
        # Reuse backbone forward but intercept before final head
        audio_emb = self.backbone.audio_encoder(waveform, attention_mask)
        text_emb = self.backbone.text_encoder(texts, waveform.device)
        fused = self.backbone.fusion_head.fuse(audio_emb, text_emb, has_text)
        # Get features before last linear
        shared = self.backbone.fusion_head.heads["phq"][:-1](fused)
        logit = self.binary_head(shared).squeeze(-1)
        return logit

# Simpler: just train with BCE loss on the existing regression output thresholded
model2 = MultimodalClinicalModel(256, 4, 8, True, True, 1).to(DEVICE)
model2.freeze_backbones_for_training()
opt2 = torch.optim.AdamW([p for p in model2.parameters() if p.requires_grad], 5e-4)

class BCELoss(nn.Module):
    def forward(self, pred, target_norm, target_raw):
        # pred is still regression output, use sigmoid(regression) as binary prob
        prob = torch.sigmoid(pred * 0.5)  # scale to make threshold meaningful
        binary_target = (target_raw >= 10).float()
        return nn.functional.binary_cross_entropy(prob, binary_target)

r2 = train_one(model2, opt2, BCELoss())
print("Exp2 Best: MAE=%.3f Sens=%.3f Spec=%.3f" % (r2[0], r2[1], r2[2]))

# ---- Experiment 3: Joint Regression + Classification ----
print("\n=== Exp3: Joint MSE + BCE ===")
model3 = MultimodalClinicalModel(256, 4, 8, True, True, 1).to(DEVICE)
model3.freeze_backbones_for_training()
opt3 = torch.optim.AdamW([p for p in model3.parameters() if p.requires_grad], 5e-4)

class JointLoss(nn.Module):
    def forward(self, pred, target_norm, target_raw):
        mse = nn.functional.mse_loss(pred, target_norm)
        prob = torch.sigmoid(pred * 0.5)
        bce = nn.functional.binary_cross_entropy(prob, (target_raw >= 10).float())
        return mse + 0.5 * bce

r3 = train_one(model3, opt3, JointLoss())
print("Exp3 Best: MAE=%.3f Sens=%.3f Spec=%.3f" % (r3[0], r3[1], r3[2]))

# ---- Summary ----
print("\n===== SUMMARY =====")
print("Exp1 WeightedMSE:      MAE=%.3f Sens=%.3f Spec=%.3f" % (r1[0], r1[1], r1[2]))
print("Exp2 BinaryClass:      MAE=%.3f Sens=%.3f Spec=%.3f" % (r2[0], r2[1], r2[2]))
print("Exp3 Joint:            MAE=%.3f Sens=%.3f Spec=%.3f" % (r3[0], r3[1], r3[2]))
