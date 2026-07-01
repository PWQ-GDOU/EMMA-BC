import sys
B = "/data/disk1/CancerEmotionSystem"
with open(B + "/phaseB/multimodal_dataset.py", "r") as f:
    lines = f.readlines()

# Find and fix the broken try/except
fixed = []
i = 0
while i < len(lines):
    line = lines[i]
    if line.strip() == "try:" and lines[i+1].strip().startswith("waveform") and lines[i+2].strip() == "except:":
        # This is our broken try/except - fix indentation
        # Original should be 12 spaces for try (inside for loop)
        prefix = "            "  # 12 spaces
        fixed.append(prefix + "try:
")
        fixed.append(prefix + "    waveform, sr = sf.read(path)
")
        fixed.append(prefix + "except:
")
        fixed.append(prefix + "    continue  # skip corrupt WAV
")
        i += 4
        continue
    fixed.append(line)
    i += 1

with open(B + "/phaseB/multimodal_dataset.py", "w") as f:
    f.writelines(fixed)

import py_compile
try:
    py_compile.compile(B + "/phaseB/multimodal_dataset.py", doraise=True)
    print("Compile OK")
except py_compile.PyCompileError as e:
    print(f"Compile: {e}")
