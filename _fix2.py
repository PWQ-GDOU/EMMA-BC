python3 << 'EOF'
import re

with open("/data/disk1/CancerEmotionSystem/phaseB/phaseB_train.py", "r") as f:
    t = f.read()

# Fix the data loading section
old = 'daic_ds = DAICWOZDataset(args.data, split="train"); modma_ds = MODMADataset(args.modma_data, args.modma_xlsx, max_audio_sec=120); full_ds = CombinedMultimodalDataset(daic_ds, modma_ds)'
new = 'daic_ds = DAICWOZDataset(args.data, split="train")
    modma_ds = MODMADataset(args.modma_data, args.modma_xlsx, max_audio_sec=120)
    full_ds = CombinedMultimodalDataset(daic_ds, modma_ds)
    train_ds, val_ds = full_ds.split_by_subject(val_ratio=args.val_split, seed=args.seed)
    _coll = collate_combined'

# Also remove the stale split_val_from_train call
# Find and remove lines that call split_val_from_train
# The pattern after our replacement: "full_ds.split_by_subject(...); _coll = collate_combined
    train_ds, val_ds = full_ds.split_val_from_train("
# We need to remove the second split call
old2 = '; _coll = collate_combined\n    train_ds, val_ds = full_ds.split_val_from_train('
new2 = ''

if old2 in t:
    t = t.replace(old2 + '\n        val_ratio=args.val_split, seed=args.seed\n    )', '')
    print("Removed stale split_val_from_train")
elif old in t:
    t = t.replace(old, new)
    # Now remove the stale second split
    t = t.replace('; _coll = collate_combined\n    train_ds, val_ds = full_ds.split_val_from_train(\n        val_ratio=args.val_split, seed=args.seed\n    )', '')
    print("Fixed data loading")
else:
    print("Pattern not found, checking manually...")
    # Find the DAICWOZDataset call
    idx = t.find('DAICWOZDataset')
    print(t[idx:idx+300])

with open("/data/disk1/CancerEmotionSystem/phaseB/phaseB_train.py", "w") as f:
    f.write(t)

import py_compile
try:
    py_compile.compile("/data/disk1/CancerEmotionSystem/phaseB/phaseB_train.py", doraise=True)
    print("Compile: OK")
except py_compile.PyCompileError as e:
    print(f"Compile: {e}")
EOF
