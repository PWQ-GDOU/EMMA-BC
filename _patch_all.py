import os, sys
B = "/data/disk1/CancerEmotionSystem"

# 1. Restore from git
os.system(f"cd {B} && git checkout -- phaseB/multimodal_dataset.py phaseB/phaseB_train.py")
print("Git restore done")

# 2. Apply ID mapping to dataset
with open(f"{B}/phaseB/multimodal_dataset.py", "r") as f:
    ds = f.read()

# Fix filter_interviewer bug
ds = ds.replace(
    "self.min_samples = int(getattr(self, 'min_audio_sec', 1.0) * sample_rate) if hasattr(self, 'min_audio_sec') else 16000",
    "self.min_samples = 16000"
)
ds = ds.replace(
    "self.min_samples = int(min_audio_sec * sample_rate)",
    "# removed — see above"
)
ds = ds.replace(
    "self.filter_interviewer = filter_interviewer",
    "self.filter_interviewer = False"
)

# Fix ID mapping
ds = ds.replace(
    'self.labels_df.set_index("subject_id", inplace=True)',
    'self.labels_df.set_index("subject_id", inplace=True)\n        self.labels_df.index = ["0" + str(sid) for sid in self.labels_df.index]'
)

# Append combined dataset code
combined = '''
class CombinedMultimodalDataset(Dataset):
    def __init__(self, daic_ds, modma_ds):
        self.daic = daic_ds; self.modma = modma_ds
        self.daic_len = len(daic_ds); self.modma_len = len(modma_ds)
        self.total = self.daic_len + self.modma_len
        print(f"[Combined] DAIC={self.daic_len} + MODMA={self.modma_len} = {self.total}")
    def __len__(self): return self.total
    def __getitem__(self, idx):
        if idx < self.daic_len:
            item = self.daic[idx]
            return {"audio":item["audio"],"texts":item["transcript"],"phq":item["phq_total"],"has_text":True,"source":"daic","pid":item["pid"]}
        else:
            item = self.modma[idx - self.daic_len]
            return {"audio":item["audio"],"texts":"","phq":item["phq"],"has_text":False,"source":"modma","pid":item["subject_id"]}
    def split_by_subject(self, val_ratio=0.15, seed=42):
        import random; rng = random.Random(seed)
        dp = list(set(s["pid"] for s in self.daic.samples)); rng.shuffle(dp)
        nv = max(1,int(len(dp)*val_ratio)); vp = set(dp[:nv])
        dt=[i for i,s in enumerate(self.daic.samples) if s["pid"] not in vp]
        dv=[i for i,s in enumerate(self.daic.samples) if s["pid"] in vp]
        sp = list(set(s["subject_id"] for s in self.modma.samples)); rng.shuffle(sp)
        nm = max(1,int(len(sp)*val_ratio)); vm = set(sp[:nm])
        mt=[i+self.daic_len for i,s in enumerate(self.modma.samples) if s["subject_id"] not in vm]
        mv=[i+self.daic_len for i,s in enumerate(self.modma.samples) if s["subject_id"] in vm]
        print(f"[Combined Split] train={len(dt)+len(mt)} (D:{len(dt)}+M:{len(mt)}), val={len(dv)+len(mv)} (D:{len(dv)}+M:{len(mv)})")
        from torch.utils.data import Subset; return Subset(self, dt+mt), Subset(self, dv+mv)

def collate_combined(batch):
    al,ml,tx,pl,ht=[],[],[],[],[]
    for item in batch:
        a=item["audio"];L=a.shape[0];al.append(a);ml.append(torch.ones(L))
        tx.append(item["texts"]);pl.append(item["phq"]);ht.append(item["has_text"])
    return {"audio":torch.stack(al),"attention_mask":torch.stack(ml),"texts":tx,"phq_total":torch.stack(pl),"has_text":torch.tensor(ht,dtype=torch.bool)}
'''
if 'class CombinedMultimodalDataset' not in ds:
    ds += "\n" + combined

with open(f"{B}/phaseB/multimodal_dataset.py", "w") as f:
    f.write(ds)
print("Dataset patched")

# 3. Patch trainer
with open(f"{B}/phaseB/phaseB_train.py", "r") as f:
    t = f.read()

# JSON fix
t = t.replace(
    'json.dump(history, f, indent=2)',
    'clean = [{k: float(v.item()) if hasattr(v,"item") else v for k,v in e.items()} for e in history]; json.dump(clean, f, indent=2)'
)

# MODMA args
t = t.replace(
    'parser.add_argument("--data", type=str, default="/data/disk1/datasets/diac_woz")',
    'parser.add_argument("--data", type=str, default="/data/disk1/datasets/diac_woz")\n    parser.add_argument("--modma_data", type=str, default="/data/disk1/datasets/modma/extracted")\n    parser.add_argument("--modma_xlsx", type=str, default="/data/disk1/datasets/modma/extracted/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx")\n    parser.add_argument("--use_modma", action="store_true", default=True)'
)

# Imports
t = t.replace(
    'from multimodal_dataset import (\n    DAICWOZDataset,\n    MODMADataset,\n    collate_daic,\n    collate_modma,\n)',
    'from multimodal_dataset import (\n    DAICWOZDataset,\n    MODMADataset,\n    CombinedMultimodalDataset,\n    collate_daic,\n    collate_modma,\n    collate_combined,\n)'
)

# Data loading
t = t.replace(
    'full_ds = DAICWOZDataset(args.data, split="train")',
    'daic_ds = DAICWOZDataset(args.data, split="train"); modma_ds = MODMADataset(args.modma_data, args.modma_xlsx, max_audio_sec=120); full_ds = CombinedMultimodalDataset(daic_ds, modma_ds)'
)
t = t.replace(
    'full_ds.split_val_from_train(',
    'full_ds.split_by_subject(val_ratio=args.val_split, seed=args.seed); _coll = collate_combined\n    train_ds, val_ds = full_ds.split_val_from_train('
)

# Fix broken data section
# The previous replace may have created broken code. Let me do targeted fixes
t = t.replace(
    'collate_fn=collate_daic, num_workers=args.num_workers,',
    'collate_fn=_coll, num_workers=args.num_workers,'
)
t = t.replace(
    'collate_fn=collate_daic, num_workers=min(2, args.num_workers),',
    'collate_fn=_coll, num_workers=min(2, args.num_workers),'
)

# has_text
t = t.replace(
    'outputs = model(audio, texts, attention_mask=attention_mask)\n        pred = outputs["phq"].squeeze(-1)',
    'ht = batch.get("has_text", None); ht = ht.to(device) if ht is not None else None\n        outputs = model(audio, texts, attention_mask=attention_mask, has_text=ht)\n        pred = outputs["phq"].squeeze(-1)'
)
t = t.replace(
    'outputs = model(audio, texts, attention_mask=attention_mask)\n        preds_norm = outputs["phq"].squeeze(-1)',
    'ht = batch.get("has_text", None); ht = ht.to(device) if ht is not None else None\n        outputs = model(audio, texts, attention_mask=attention_mask, has_text=ht)\n        preds_norm = outputs["phq"].squeeze(-1)'
)

# num_workers default
t = t.replace('default=2,', 'default=0,')

with open(f"{B}/phaseB/phaseB_train.py", "w") as f:
    f.write(t)
print("Trainer patched")

# Verify
import py_compile
for fn in ["phaseB/multimodal_dataset.py", "phaseB/phaseB_train.py"]:
    try:
        py_compile.compile(f"{B}/{fn}", doraise=True)
        print(f"Compile {fn}: OK")
    except py_compile.PyCompileError as e:
        print(f"Compile {fn}: FAIL — {e}")
