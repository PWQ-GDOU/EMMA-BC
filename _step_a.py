# Step 1: Rename corrupt WAV
import os
bad = "/data/disk1/datasets/modma/extracted/audio_lanzhou_2015/02010004/24.wav"
if os.path.exists(bad):
    os.rename(bad, bad + ".bak")
    print("Renamed corrupt WAV")
else:
    print("File already renamed or not found")

# Step 2: Fix collate_combined
B = "/data/disk1/CancerEmotionSystem"
with open(B + "/phaseB/multimodal_dataset.py", "r") as f:
    ds = f.read()

old = """def collate_combined(batch):
    al,ml,tx,pl,ht=[],[],[],[],[]
    for item in batch:
        a=item["audio"];L=a.shape[0];al.append(a);ml.append(torch.ones(L))
        tx.append(item["texts"]);pl.append(item["phq"]);ht.append(item["has_text"])
    return {"audio":torch.stack(al),"attention_mask":torch.stack(ml),"texts":tx,"phq_total":torch.stack(pl),"has_text":torch.tensor(ht,dtype=torch.bool)}"""

new = """def collate_combined(batch):
    max_len = max(item["audio"].shape[0] for item in batch)
    al, ml, tx, pl, ht = [], [], [], [], []
    for item in batch:
        a = item["audio"]; L = a.shape[0]
        if L < max_len:
            a = torch.cat([a, torch.zeros(max_len - L)])
            m = torch.cat([torch.ones(L), torch.zeros(max_len - L)])
        else:
            a = a[:max_len]; m = torch.ones(max_len)
        al.append(a); ml.append(m)
        tx.append(item["texts"]); pl.append(item["phq"]); ht.append(item["has_text"])
    return {"audio":torch.stack(al),"attention_mask":torch.stack(ml),"texts":tx,"phq_total":torch.stack(pl),"has_text":torch.tensor(ht,dtype=torch.bool)}"""

if old in ds:
    ds = ds.replace(old, new)
    with open(B + "/phaseB/multimodal_dataset.py", "w") as f:
        f.write(ds)
    print("collate_combined: padded version applied")
else:
    print("WARNING: old collate_combined not found!")
    # Find what's actually there
    idx = ds.find("def collate_combined")
    print(ds[idx:idx+400])

# Verify
import py_compile
try:
    py_compile.compile(B + "/phaseB/multimodal_dataset.py", doraise=True)
    print("Compile: OK")
except Exception as e:
    print(f"Compile FAIL: {e}")
