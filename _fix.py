import sys
B = "/data/disk1/CancerEmotionSystem"

# Fix 1: num_workers
with open(B + "/phaseB/phaseB_train.py", "r") as f:
    t = f.read()
t = t.replace('default=2,', 'default=0,')
t = t.replace('num_workers=min(2, args.num_workers)', 'num_workers=args.num_workers')
with open(B + "/phaseB/phaseB_train.py", "w") as f:
    f.write(t)
print("num_workers: OK")

# Fix 2: WAV error handling in MODMA  
with open(B + "/phaseB/multimodal_dataset.py", "r") as f:
    ds = f.read()

old = "waveform, sr = sf.read(path)"
new = "try:\n                waveform, sr = sf.read(path)\n            except:\n                continue"
if old in ds:
    ds = ds.replace(old, new)
    with open(B + "/phaseB/multimodal_dataset.py", "w") as f:
        f.write(ds)
    print("WAV fix: OK")
else:
    print("WAV fix: pattern not found")

# Verify
import py_compile
for fn in ["phaseB/multimodal_dataset.py", "phaseB/phaseB_train.py"]:
    try:
        py_compile.compile(B + "/" + fn, doraise=True)
        print(f"Compile {fn}: OK")
    except Exception as e:
        print(f"Compile {fn}: {e}")
