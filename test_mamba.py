import sys; sys.path.insert(0, '/data/disk3/EMMA-BC')
from mamba_encoder import MambaEncoder, USE_FAST_MAMBA
import torch
print(f"MambaEncoder OK, fast={USE_FAST_MAMBA}")
m = MambaEncoder(d_model=256, n_layers=4)
x = torch.randn(2, 100, 256)
y = m(x)
print(f"Shape: {x.shape} -> {y.shape}, Params: {sum(p.numel() for p in m.parameters()):,}")
print("TEST PASSED")
