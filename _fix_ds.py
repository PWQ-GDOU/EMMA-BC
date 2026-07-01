#!/usr/bin/env python3
"""Fix indentation bug in multimodal_dataset.py"""
import py_compile

path = "/data/disk1/CancerEmotionSystem/phaseB/multimodal_dataset.py"

with open(path, 'r') as f:
    content = f.read()

# 1. Remove wrongly-indented split_by_subject from inside collate_modma
# Find: "subject_ids": sids,\n    }\n        """  (closing brace of collate_modma,
# followed by indented docstring from old split_by_subject)
bad_marker = '        "subject_ids": sids,\n    }\n        """'
pos = content.find(bad_marker)
if pos >= 0:
    # Find end of bad section: next top-level definition
    rest = content[pos + len('        "subject_ids": sids,\n    }\n'):]
    # Find next "\n\ndef " or "\n\nclass " or "\n\n#"
    next_def = len(rest)
    for marker in ['\n\ndef ', '\n\nclass ', '\n\n# =====']:
        idx = rest.find(marker)
        if idx >= 0 and idx < next_def:
            next_def = idx
    
    # Reconstruct: keep everything before bad section + clean closing brace + rest 
    clean_end = rest[next_def:]
    content = content[:pos] + '        "subject_ids": sids,\n    }' + clean_end
    print("Fixed: removed bad split_by_subject from inside collate_modma")

# 2. Add proper split_by_subject method (as class method, before collate_modma)
if '    def split_by_subject(self, val_ratio=0.15, test_ratio=0.15, seed=42):' not in content:
    split_method = '''    def split_by_subject(self, val_ratio=0.15, test_ratio=0.15, seed=42):
        """Split MODMA data by SUBJECT to prevent data leakage."""
        import random
        rng = random.Random(seed)
        subject_ids = sorted(set(s["subject_id"] for s in self.samples))
        rng.shuffle(subject_ids)
        n_val = max(1, int(len(subject_ids) * val_ratio))
        val_ids = set(subject_ids[:n_val])
        train_ids = set(subject_ids[n_val:])
        train_idx = [i for i, s in enumerate(self.samples) if s["subject_id"] in train_ids]
        val_idx = [i for i, s in enumerate(self.samples) if s["subject_id"] in val_ids]
        print(f"[MODMA Subject Split] train={len(train_ids)} ({len(train_idx)}), val={len(val_ids)} ({len(val_idx)})")
        from torch.utils.data import Subset
        return Subset(self, train_idx), Subset(self, val_idx)

'''
    content = content.replace('\n\ndef collate_modma(batch):', '\n\n' + split_method + '\ndef collate_modma(batch):')
    print("Added proper split_by_subject method")

# 3. Write back
with open(path, 'w') as f:
    f.write(content)

# 4. Verify
try:
    py_compile.compile(path, doraise=True)
    print("Compile: OK")
except py_compile.PyCompileError as e:
    print(f"Compile error: {e}")

# 5. Check combined dataset code exists
with open(path, 'r') as f:
    content = f.read()
for marker in ['class CombinedMultimodalDataset', 'def collate_combined', 'split_by_subject']:
    count = content.count(marker)
    print(f"  {marker}: {count} occurrences")
