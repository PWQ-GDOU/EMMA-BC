import sys
B = "/data/disk1/CancerEmotionSystem"
with open(B + "/phaseB/multimodal_dataset.py", "r") as f:
    ds = f.read()

# Fix 1: Replace MODMA _load_audio to pad properly
old_load = """        if combined.shape[0] > self.max_samples:
            combined = combined[:self.max_samples]
        elif combined.shape[0] < self.max_samples:
            combined = torch.cat([
                combined,
                torch.zeros(self.max_samples - combined.shape[0])
            ])"""

# This should already exist - check if it's there  
if old_load in ds:
    print("MODMA _load_audio already has padding")
else:
    print("MODMA _load_audio padding NOT found - checking...")
    # Find the actual code
    idx = ds.find("if combined.shape[0] > self.max_samples:")
    print(ds[idx:idx+300])

# Fix 2: Replace collate_combined with padded version
old_coll = "def collate_combined(batch):"
idx = ds.find(old_coll)
if idx >= 0:
    # Find end of function (next blank line followed by non-whitespace)
    rest = ds[idx:]
    end_markers = ["\n\ndef ", "\n\nclass ", "\n\n# "]
    end = len(rest)
    for m in end_markers:
        p = rest.find(m, 20)
        if 0 < p < end:
            end = p
    
    new_coll = '''def collate_combined(batch):
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
    return {"audio":torch.stack(al),"attention_mask":torch.stack(ml),"texts":tx,"phq_total":torch.stack(pl),"has_text":torch.tensor(ht,dtype=torch.bool)}
'''
    ds = ds[:idx] + new_coll + ds[idx+end:]
    
    with open(B + "/phaseB/multimodal_dataset.py", "w") as f:
        f.write(ds)
    
    import py_compile
    try:
        py_compile.compile(B + "/phaseB/multimodal_dataset.py", doraise=True)
        print("Compile OK - collate_combined patched")
    except Exception as e:
        print(f"Compile FAIL: {e}")
else:
    print("collate_combined not found!")
