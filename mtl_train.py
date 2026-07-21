#!/usr/bin/env python3
"""Multi-task Mamba training with logging."""
import os, sys
sys.path.insert(0, "/data/disk3/EMMA-BC")
sys.path.insert(0, "/data/disk3/EMMA-BC/phaseB")

LOG = open("/data/disk3/EMMA-BC/mtl_result.log", "w")
def log(msg):
    print(msg); LOG.write(str(msg) + "\n"); LOG.flush()

import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from multimodal_dataset import DAICWOZDataset, MODMADataset, CombinedMultimodalDataset, collate_combined
from multimodal_model import MultimodalClinicalModel, set_seed

DEVICE = torch.device("cuda:0"); set_seed(42)
log("Device: " + str(DEVICE))

# Data
log("Loading data...")
daic = DAICWOZDataset("/data/disk1/datasets/diac_woz", "train", False, 1.0, 5)
modma = MODMADataset("/data/disk1/datasets/modma/extracted",
    "/data/disk1/datasets/modma/extracted/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx", 120)
full = CombinedMultimodalDataset(daic, modma)
train_ds, val_ds = full.split_by_subject(0.15, 42)
log("Train: " + str(len(train_ds)) + " Val: " + str(len(val_ds)))

# Collate with multi-task labels
def collate_mtl(batch):
    ml = max((i["audio"].shape[0] for i in batch if i["audio"].shape[0] > 0), default=16000)
    al, mk, ph, ga, ps = [], [], [], [], []
    for it in batch:
        a = it["audio"]; L = a.shape[0]
        if L < 160:  # skip < 10ms audio
            a = torch.zeros(ml)
        al.append(torch.cat([a[:ml], torch.zeros(max(0,ml-L))]))
        mk.append(torch.cat([torch.ones(min(L,ml)), torch.zeros(max(0,ml-L))]))
        ph.append(it.get("phq_total", it.get("phq", torch.tensor(float('nan')))).float())
        gv = it.get("gad", None); ps.append(torch.tensor(gv).float() if gv is not None else torch.tensor(float('nan')))
        pv = it.get("psqi", None); ga.append(torch.tensor(pv).float() if pv is not None else torch.tensor(float('nan')))
    return {"audio": torch.stack(al), "attention_mask": torch.stack(mk),
            "phq": torch.stack(ph), "gad": torch.stack(ga), "psqi": torch.stack(ps),
            "has_text": torch.zeros(len(batch), dtype=torch.bool)}

# Simple loader (skip oversampling for speed)
gen = torch.Generator().manual_seed(42)
tl = DataLoader(train_ds, 1, True, collate_fn=collate_mtl, num_workers=0, pin_memory=True, generator=gen, drop_last=True)
vl = DataLoader(val_ds, 1, False, collate_fn=collate_mtl, num_workers=0, pin_memory=True)

# Model
log("Building 3-task model...")
model = MultimodalClinicalModel(256, 4, 8, True, True, 3).to(DEVICE)
model.freeze_backbones_for_training()
log("Trainable: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], 5e-4)
phq_mean, phq_std = torch.tensor(8.0), torch.tensor(7.0)  # Pre-computed from earlier runs
log("PHQ mean=" + str(round(phq_mean.item(),2)) + " std=" + str(round(phq_std.item(),2)))

# Train
best_mae = float("inf")
for ep in range(1, 51):
    model.train()
    for batch in tqdm(tl, desc="E"+str(ep)+"/50", leave=False):
        a = batch["audio"].to(DEVICE); m = batch["attention_mask"].to(DEVICE)
        phq = batch["phq"].to(DEVICE); gad = batch["gad"].to(DEVICE); psqi = batch["psqi"].to(DEVICE)
        out = model(a, [""], attention_mask=m, has_text=torch.zeros(a.shape[0], dtype=torch.bool).to(DEVICE))
        loss = torch.tensor(0.0, device=DEVICE); n = 0
        
        pm = ~torch.isnan(phq)
        if pm.any():
            loss = loss + nn.functional.mse_loss(out["phq"].squeeze(-1)[pm], ((phq[pm]-phq_mean)/phq_std)); n += 1
        gm = ~torch.isnan(gad)
        if gm.any():
            t = gad[gm]; loss = loss + nn.functional.mse_loss(out["gad"].squeeze(-1)[gm], (t-t.mean())/(t.std().clamp(min=1e-6))); n += 1
        sm = ~torch.isnan(psqi)
        if sm.any():
            t = psqi[sm]; loss = loss + nn.functional.mse_loss(out["psqi"].squeeze(-1)[sm], (t-t.mean())/(t.std().clamp(min=1e-6))); n += 1
        
        if n > 0:
            (loss/n).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
    
    if ep % 5 == 0:
        model.eval(); pp, pl = [], []
        with torch.no_grad():
            for batch in vl:
                a = batch["audio"].to(DEVICE); m = batch["attention_mask"].to(DEVICE)
                phq = batch["phq"].to(DEVICE)
                out = model(a, [""], attention_mask=m, has_text=torch.zeros(a.shape[0], dtype=torch.bool).to(DEVICE))
                mk = ~torch.isnan(phq)
                if mk.any():
                    pp.append((out["phq"].squeeze(-1)[mk]*phq_std.to(DEVICE)+phq_mean.to(DEVICE)).cpu())
                    pl.append(phq[mk].cpu())
        if pp:
            pr = torch.cat(pp); lb = torch.cat(pl); mae = (pr-lb).abs().mean().item()
            pb = (pr>=10).int(); lb2 = (lb>=10).int()
            tp = ((pb==1)&(lb2==1)).sum().item(); tn = ((pb==0)&(lb2==0)).sum().item()
            fp = ((pb==1)&(lb2==0)).sum().item(); fn = ((pb==0)&(lb2==1)).sum().item()
            s = tp/(tp+fn+1e-8); sp = tn/(tn+fp+1e-8)
            msg = "E%02d MAE=%.3f Sens=%.3f Spec=%.3f" % (ep, mae, s, sp)
            if mae < best_mae: best_mae = mae; msg += " <- BEST"
            log(msg)

log("Done. Best MAE: " + str(round(best_mae, 3)))
LOG.close()
