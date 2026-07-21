"""
Mamba-based Audio Encoder for EMMA-BC Depression Detection.
Replaces the 4-layer Transformer encoder with 4 Mamba blocks.

Two options:
1. Production: pip install mamba-ssm (needs CUDA, uses optimized selective_scan)
2. Fallback: pure PyTorch implementation below (CPU-compatible, for testing)

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with 
           Selective State Spaces", NeurIPS 2023
"""

import torch
import torch.nn as nn
import math


# ============================================================
# OPTION 1: Production Mamba (install mamba-ssm first)
# ============================================================
try:
    from mamba_ssm import Mamba
    
    class FastMambaBlock(nn.Module):
        """Thin wrapper over official mamba-ssm."""
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            residual = x
            x = self.norm(x)
            x = self.mamba(x)
            x = self.dropout(x)
            return x + residual
    
    USE_FAST_MAMBA = True
    print("[Mamba] Using mamba-ssm (CUDA optimized)")

except ImportError:
    USE_FAST_MAMBA = False
    print("[Mamba] mamba-ssm not found, using pure PyTorch fallback")


# ============================================================
# OPTION 2: Pure PyTorch Fallback (CPU-compatible)
# ============================================================
if not USE_FAST_MAMBA:
    
    class MambaBlock(nn.Module):
        """Pure PyTorch Mamba implementing selective state space model."""
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.d_inner = int(d_model * expand)
            self.d_state = d_state
            
            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
            
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner, out_channels=self.d_inner,
                kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner,
            )
            
            self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
            self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)
            
            A = torch.arange(1, d_state + 1).float().unsqueeze(0)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))
            
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            residual = x
            B, L, D = x.shape
            x_norm = self.norm(x)
            
            xz = self.in_proj(x_norm)
            x_proj, z = xz.chunk(2, dim=-1)
            
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :L]
            x_conv = nn.functional.silu(x_conv).transpose(1, 2)
            
            # Selective SSM (sequential — use mamba-ssm for speed)
            proj = self.x_proj(x_conv)
            dt, B_ssm, C_ssm = torch.split(
                proj, [self.d_inner, self.d_state, self.d_state], dim=-1
            )
            dt = nn.functional.softplus(self.dt_proj(x_conv) + dt)
            A = -torch.exp(self.A_log)
            
            h = torch.zeros(B, self.d_inner, self.d_state, device=x_conv.device, dtype=x_conv.dtype)
            outputs = []
            for t in range(L):
                dt_t = dt[:, t]
                A_bar = torch.exp(dt_t.unsqueeze(-1) * A)
                B_t = B_ssm[:, t].unsqueeze(1)
                u_t = x_conv[:, t].unsqueeze(-1)
                C_t = C_ssm[:, t].unsqueeze(1)
                h = h * A_bar + B_t * u_t
                y = (h * C_t).sum(dim=-1) + x_conv[:, t] * self.D
                outputs.append(y)
            
            y = torch.stack(outputs, dim=1)
            z = nn.functional.silu(z)
            y = y * z
            out = self.out_proj(y)
            out = self.dropout(out)
            return out + residual


# ============================================================
# Mamba Encoder
# ============================================================
class MambaEncoder(nn.Module):
    """Direct replacement for nn.TransformerEncoder."""
    def __init__(self, d_model=256, n_layers=4, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        Block = FastMambaBlock if USE_FAST_MAMBA else MambaBlock
        self.layers = nn.ModuleList([
            Block(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_final = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x)
        return self.norm_final(x)


def can_use_cuda_mamba():
    return USE_FAST_MAMBA


# ============================================================
# Integration guide
# ============================================================
if __name__ == "__main__":
    B, L, D = 4, 300, 256
    x = torch.randn(B, L, D)
    
    encoder = MambaEncoder(d_model=D, n_layers=4)
    out = encoder(x)
    
    n_p = sum(p.numel() for p in encoder.parameters())
    
    tf = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=D, nhead=8, dim_feedforward=1024,
                                    dropout=0.1, activation='gelu', batch_first=True),
        num_layers=4,
    )
    n_tf = sum(p.numel() for p in tf.parameters())
    
    print(f"Input:  {x.shape} -> Output: {out.shape}")
    print(f"Transformer params: {n_tf:,}")
    print(f"Mamba params:      {n_p:,}")
    print(f"Reduction:         {n_tf-n_p:,} ({(n_tf-n_p)/n_tf*100:.1f}%)")
    print(f"\nTo use: copy MambaEncoder into train_fusion.py and replace TransformerEncoder.")
