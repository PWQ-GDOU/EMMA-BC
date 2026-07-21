#!/usr/bin/env python3
"""Layer 3 validation: Train with synthetic data and compare."""
import os, sys, glob
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from tqdm import tqdm

sys.path.insert(0, "/data/disk3/EMMA-BC")
sys.path.insert(0, "/data/disk3/EMMA-BC/phaseB")
from multimodal_dataset import DAICWOZDataset, MODMADataset, CombinedMultimodalDataset, collate_combined
from multimodal_model import MultimodalClinicalModel, set_seed

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Load synthetic
SYNTH = "/data/disk3/EMMA-BC/synthetic"
synth_audios = []
for f in sorted(glob.glob(SYNTH + "/synth_*.wav")):
    y, sr = sf.read(f)
    if sr != 16000:
        import librosa
        y = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
    synth_audios.append(torch.from_numpy(y).float())
print("Synth samples:", len(synth_audios))

class SynthDS(Dataset):
    def __init__(self, audios): self.a = audios
    def __len__(self): return len(self.a)
    def __getitem__(self, i):
        return {"audio": self.a[i], "texts": "", "phq_total": torch.tensor(15.0), "has_text": torch.tensor(False)}

def coll_synth(batch):
    ml = max(i["audio"].shape[0] for i in batch)
    al, ml2, pl = [], [], []
    for i in batch:
        a = i["audio"]; L = a.shape[0]
        if L < ml: a = torch.cat([a, torch.zeros(ml-L)])
        else: a = a[:ml]
        al.append(a)
        ml2.append(torch.cat([torch.ones(min(L,ml)), torch.zeros(max(0,ml-L))]))
        pl.append(i["phq_total"])
    return {"audio": torch.stack(al), "attention_mask": torch.stack(ml2), "texts": [""]*len(batch), "phq_total": torch.stack(pl), "has_text": torch.zeros(len(batch), dtype=torch.bool)}

print("Loading datasets...")
set_seed(42)
daic = DAICWOZDataset("/data/disk1/datasets/diac_woz", split="train", filter_interviewer=False, min_audio_sec=1.0, max_audio_sec=3)
modma = MODMADataset("/data/disk1/datasets/modma/extracted", "/data/disk1/datasets/modma/extracted/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx", max_audio_sec=120)
full = CombinedMultimodalDataset(daic, modma)
train_ds, val_ds = full.split_by_subject(val_ratio=0.15, seed=42)
print("Train:", len(train_ds), "Val:", len(val_ds))

# Oversampling
tmp_ldr = DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=collate_combined, num_workers=0, drop_last=False)
all_l = []
for b in tmp_ldr: all_l.append(b["phq_total"])
all_l = torch.cat(all_l)
pos = (all_l >= 10.0).sum().item(); neg = (~(all_l >= 10.0)).sum().item()
print("Pos:", pos, "Neg:", neg)
if pos > 0 and neg > 0:
    w = torch.where(all_l >= 10.0, neg/pos*2.0, 1.0)
    sampler = WeightedRandomSampler(w.tolist(), int(len(w)*1.3), replacement=True)
else:
    sampler = None

gen = torch.Generator().manual_seed(42)
train_loader = DataLoader(train_ds, batch_size=1, sampler=sampler, shuffle=(sampler is None), collate_fn=collate_combined, num_workers=0, pin_memory=True, generator=gen, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_combined, num_workers=0, pin_memory=True)

# Normalizer
mn = all_l.mean(); sd = all_l.std().clamp(min=1e-6)
def norm(x): return (x-mn.to(x.device))/sd.to(x.device)
def denorm(x): return x*sd.to(x.device)+mn.to(x.device)

print("Building model...")
model = MultimodalClinicalModel(d_model=256, n_layers=4, n_heads=8, freeze_audio_w2v=True, freeze_text_bert=True, n_tasks=1).to(DEVICE)
model.freeze_backbones_for_training()
print("Trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
crit = nn.MSELoss()
synth_ds = SynthDS(synth_audios)
best_mae = float("inf")

for ep in range(1, 26):
    model.train()
    for batch in tqdm(train_loader, desc="E"+str(ep)+"/25", leave=False):
        a = batch["audio"].to(DEVICE)
        m = batch["attention_mask"].to(DEVICE)
        l = batch["phq_total"].to(DEVICE)
        si = torch.randint(0, len(synth_ds), (1,))
        sb = coll_synth([synth_ds[int(i)] for i in si])
        sa = sb["audio"].to(DEVICE)
        sm = sb["attention_mask"].to(DEVICE)
        # Pad or truncate to match batch length
        if sa.shape[1] > a.shape[1]:
            sa = sa[:, :a.shape[1]]
            sm = sm[:, :a.shape[1]]
        elif sa.shape[1] < a.shape[1]:
            pad = torch.zeros(sa.shape[0], a.shape[1] - sa.shape[1], device=DEVICE)
            sa = torch.cat([sa, pad], dim=1)
            sm = torch.cat([sm, pad], dim=1)
        a = torch.cat([a, sa], dim=0)
        m = torch.cat([m, sm], dim=0)
        l = torch.cat([l, sb["phq_total"].to(DEVICE)], dim=0)
        ln = norm(l.unsqueeze(-1)).squeeze(-1)
        opt.zero_grad()
        out = model(a, [""] * a.shape[0], attention_mask=m, has_text=torch.zeros(a.shape[0], dtype=torch.bool).to(DEVICE))
        loss = crit(out["phq"].squeeze(-1), ln)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    ap, al2 = [], []
    with torch.no_grad():
        for batch in val_loader:
            a = batch["audio"].to(DEVICE)
            m = batch["attention_mask"].to(DEVICE)
            l = batch["phq_total"].to(DEVICE)
            out = model(a, [""] * a.shape[0], attention_mask=m, has_text=torch.zeros(a.shape[0], dtype=torch.bool).to(DEVICE))
            p = denorm(out["phq"].squeeze(-1))
            ap.append(p.cpu()); al2.append(l.cpu())
    preds = torch.cat(ap); lbls = torch.cat(al2)
    mae = (preds-lbls).abs().mean().item()
    pb = (preds>=10.0).int(); lb = (lbls>=10.0).int()
    tp = ((pb==1)&(lb==1)).sum().item(); tn = ((pb==0)&(lb==0)).sum().item()
    fp = ((pb==1)&(lb==0)).sum().item(); fn = ((pb==0)&(lb==1)).sum().item()
    sens = tp/(tp+fn+1e-8); spec = tn/(tn+fp+1e-8)
    msg = "E"+str(ep).zfill(2)+" MAE="+str(round(mae,3))+" Sens="+str(round(sens,3))+" Spec="+str(round(spec,3))
    if mae < best_mae: best_mae = mae; msg += " <- BEST"
    print(msg)

print("Done. Best MAE: " + str(round(best_mae,3)))
