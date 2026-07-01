# Fix MODMA _load_audio: ONLY the one in MODMADataset (inside for loop)
B = "/data/disk1/CancerEmotionSystem"
with open(B + "/phaseB/multimodal_dataset.py", "r") as f:
    ds = f.read()

# Find all sf.read(path) occurrences
import re
for m in re.finditer(r'sf\.read\(path\)', ds):
    start = max(0, m.start() - 80)
    end = min(len(ds), m.end() + 80)
    ctx = ds[start:end].replace('\n', '\\n')
    print(f"  pos={m.start()}: ...{ctx}...")

# The DAIC-WOZ one (8-space indent) vs MODMA one (16-space indent)
# Let me check which has 16 spaces
print("\nDAIC-WOZ context:")
for i, l in enumerate(ds.splitlines(), 1):
    if 'sf.read(path)' in l:
        print(f"  Line {i}: [{len(l)-len(l.lstrip())} spaces] {l.strip()}")

# Replace ONLY the one with 16 spaces indent (inside for loop in MODMA)
lines = ds.splitlines()
new_lines = []
for i, line in enumerate(lines):
    if 'sf.read(path)' in line and len(line) - len(line.lstrip()) >= 16:
        # This is the MODMA one — add try/except
        indent = ' ' * 16
        new_lines.append(f"{indent}try:")
        new_lines.append(f"{indent}    waveform, sr = sf.read(path)")
        new_lines.append(f"{indent}except:")
        new_lines.append(f"{indent}    continue  # skip corrupt WAV")
    else:
        new_lines.append(line)

ds = '\n'.join(new_lines)
with open(B + "/phaseB/multimodal_dataset.py", "w") as f:
    f.write(ds)

import py_compile
try:
    py_compile.compile(B + "/phaseB/multimodal_dataset.py", doraise=True)
    print("Compile OK")
except Exception as e:
    print(f"Compile: {e}")

# Verify
for i, l in enumerate(ds.splitlines(), 1):
    if 'sf.read' in l or 'try:' in l or 'except:' in l:
        if i > 240:
            print(f"Line {i}: {l}")
