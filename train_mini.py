import os, sys, json, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from pathlib import Path
import soundfile as sf, pandas as pd
from tqdm import tqdm

BASE = Path("/data/disk1/CancerEmotionSystem")
sys.path.insert(0, str(BASE))
from src.models.wavlm_adapter import DepressionClassifier

DEVICE = torch.device("cuda:0")
SR, MAX_SEC = 16000, 30
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def load_data(split):
    data_dir = Path("/data/disk1/datasets/diac_woz")
    df = pd.read_csv(data_dir / "labels" / f"{split}_split.csv")
    samples = []
    for _, row in df.iterrows():
        pid = row["Participant_ID"]
        ap = data_dir / f"{pid}_P" / f"{pid}_AUDIO.wav"
        if not ap.exists(): continue
        audio_raw, _ = sf.read(str(ap))
        audio = torch.FloatTensor(audio_raw)
        max_s = int(MAX_SEC * SR)
        if len(audio) > max_s: audio = audio[:max_s]
        else: audio = torch.cat([audio, torch.zeros(max_s - len(audio))])
        tp = data_dir / f"{pid}_P" / f"{pid}_Transcript.csv"
        text = ""
        if tp.exists():
            try:
                tdf = pd.read_csv(tp)
                text = " ".join(tdf["Text"].dropna().tolist())
            except: pass
        import math; label = float(row.get("PHQ_Score)))
        samples.append({"pid": pid, "audio": audio, "text": text, "label": label})
    return samples

def collate(batch):
    audios = [b["audio"] for b in batch]
    max_l = max(a.shape[0] for a in audios)
    padded = torch.zeros(len(audios), max_l)
    mask = torch.zeros(len(audios), max_l)
    for i, a in enumerate(audios): padded[i,:a.shape[0]] = a; mask[i,:a.shape[0]] = 1
    texts = [b["text"] for b in batch]
    labels = torch.FloatTensor([b["label"] for b in batch])
    return padded, mask, texts, labels

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels, probs = [], [], []
    for audio, amask, texts, lbls in loader:
        audio, amask, lbls = audio.to(DEVICE), amask.to(DEVICE), lbls.to(DEVICE)
        logits = model(audio, texts, None, amask, None).squeeze(-1)
        preds.extend( (logits >= 10).long().cpu().tolist())
        labels.extend(lbls.cpu().tolist())
        probs.extend(torch.sigmoid((logits-10)/3).cpu().tolist())
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    return {"acc": accuracy_score(labels,preds), "f1": f1_score(labels,preds,zero_division=0),
            "auc": roc_auc_score(labels,probs) if len(set(labels))>1 else 0.5,
            "sens": tp/(tp+fn+1e-8), "spec": tn/(tn+fp+1e-8)}

print("Loading data...")
train = load_data("train")
dev = load_data("dev")
print(f"Train: {len(train)}, Dev: {len(dev)}")

tr_dl = DataLoader(train, 1, shuffle=True, collate_fn=collate, drop_last=True)
te_dl = DataLoader(dev, 1, collate_fn=collate)

model = DepressionClassifier().to(DEVICE)
print(f"Trainable: {model.trainable_params():,}")

lbs = [s["label"] for s in train]
pos = sum(lbs); neg = len(lbs) - pos
crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, neg*3/max(pos,1)], device=DEVICE))
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

best_f1, patience, state = 0, 0, None
for ep in range(1, 31):
    model.train()
    for audio, amask, texts, labels in tr_dl:
        audio, amask, labels = audio.to(DEVICE), amask.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(audio, texts, None, amask, None).squeeze(-1), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    vm = evaluate(model, te_dl)
    if vm["f1"] > best_f1: best_f1 = vm["f1"]; patience = 0; state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    else: patience += 1
    if ep % 5 == 0: print(f"Ep{ep}: f1={vm["f1"]:.3f} auc={vm["auc"]:.3f} loss={loss.item():.3f}")
    if patience >= 15: break

if state: model.load_state_dict(state)
final = evaluate(model, te_dl)
print(f"Final: Acc={final["acc"]:.3f} F1={final["f1"]:.3f} AUC={final["auc"]:.3f} Sens={final["sens"]:.3f} Spec={final["spec"]:.3f}")
json.dump(final, open("results/wavlm_first.json", "w"), indent=2)
print("Done!")
