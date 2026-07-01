import sys
B = "/data/disk1/CancerEmotionSystem"
with open(f"{B}/phaseB/phaseB_train.py", "r") as f:
    t = f.read()

# Find and replace the broken data loading
# Current: split_val_from_train being called on CombinedMultimodalDataset
# Fix: replace the entire block from "daic_ds" to the end of split call

lines = t.split("\n")
new_lines = []
skip_until = None
i = 0
while i < len(lines):
    line = lines[i]
    if "daic_ds = DAICWOZDataset" in line or "DAICWOZDataset(args.data" in line:
        # Replace this section
        new_lines.append("    daic_ds = DAICWOZDataset(args.data, split=\"train\")")
        new_lines.append("    modma_ds = MODMADataset(args.modma_data, args.modma_xlsx, max_audio_sec=120)")
        new_lines.append("    full_ds = CombinedMultimodalDataset(daic_ds, modma_ds)")
        new_lines.append("    train_ds, val_ds = full_ds.split_by_subject(val_ratio=args.val_split, seed=args.seed)")
        new_lines.append("    _coll = collate_combined")
        # Skip lines until we find the next section (DataLoader creation)
        while i < len(lines) and "DataLoader(" not in lines[i]:
            i += 1
        # Also skip lines with split_val_from_train in them
        i -= 1  # Back up to include the DataLoader line
        continue
    new_lines.append(line)
    i += 1

with open(f"{B}/phaseB/phaseB_train.py", "w") as f:
    f.write("\n".join(new_lines))

import py_compile
try:
    py_compile.compile(f"{B}/phaseB/phaseB_train.py", doraise=True)
    print("Compile OK")
except py_compile.PyCompileError as e:
    print(f"Compile: {e}")
