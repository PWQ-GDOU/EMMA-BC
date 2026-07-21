#!/bin/bash
set -e
VENV=/data/disk3/EMMA-BC/venv/bin/pip

echo "=== Installing PyTorch CUDA 12.6 ==="
$VENV install torch torchaudio --index-url https://download.pytorch.org/whl/cu126 -q

echo "=== Installing core deps ==="
$VENV install transformers tokenizers soundfile librosa numpy pandas tqdm PyYAML openpyxl -q

echo "=== Installing mamba-ssm ==="
$VENV install mamba-ssm causal-conv1d -q 2>&1 || echo "(mamba-ssm unavailable, will use pure PyTorch fallback)"

echo "=== Verify ==="
/data/disk3/EMMA-BC/venv/bin/python -c "
import torch; print(f'Torch: {torch.__version__}')
import transformers; print(f'Transformers: {transformers.__version__}')
print(f'GPU count: {torch.cuda.device_count()}')
try:
    from mamba_ssm import Mamba
    print('mamba-ssm: OK')
except:
    print('mamba-ssm: fallback mode')
"
echo "=== DONE ==="

