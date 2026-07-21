"""
Supervised Contrastive Loss + MoCo Memory Queue.
For EMMA-BC Phase B training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoQueue:
    """Momentum Contrast queue: stores past embeddings as negative samples."""
    
    def __init__(self, dim=64, K=256):
        self.K = K
        self.queue = F.normalize(torch.randn(K, dim), dim=-1)
        self.ptr = 0
        self.full = False
    
    @torch.no_grad()
    def enqueue(self, keys):
        """keys: [B, dim] — normalized embeddings from current batch."""
        B = keys.shape[0]
        for i in range(B):
            self.queue[self.ptr] = keys[i]
            self.ptr = (self.ptr + 1) % self.K
        if self.ptr == 0:
            self.full = True
    
    def get(self):
        """Return current valid portion of the queue [n, dim]."""
        end = self.K if self.full else self.ptr
        return self.queue[:end].clone()
    
    def size(self):
        return self.K if self.full else self.ptr


def supcon_loss(features, phq_labels, queue, temperature=0.07):
    """
    Supervised contrastive loss.
    
    Args:
        features: [B, dim] — normalized contrast embeddings of current batch
        phq_labels: [B] — PHQ total scores (>=10 = depressed group)
        queue: [K, dim] — MoCo queue (all treated as negatives)
        temperature: softmax temperature (default 0.07)
    
    Returns:
        scalar loss
    """
    B, D = features.shape
    device = features.device
    
    # Binary grouping: depressed (1) vs non-depressed (0)
    is_dep = (phq_labels >= 10.0).float()  # [B]
    
    # ---- Build similarity matrix ----
    # Positive mask within batch: same class, excluding self
    pos_mask = (is_dep.unsqueeze(0) == is_dep.unsqueeze(1)).float()  # [B, B]
    pos_mask.fill_diagonal_(0.0)
    
    # Concatenate batch features with queue
    all_feat = torch.cat([features, queue.to(device)], dim=0)  # [B+K, D]
    
    # Similarity
    sim = torch.matmul(features, all_feat.T) / temperature  # [B, B+K]
    
    # Numerical stability: subtract per-row max
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    sim = sim - sim_max
    
    # Positive mask extended to B+K: queue samples are ALL negatives
    K = queue.shape[0]
    pos_mask_full = torch.zeros(B, B + K, device=device)
    pos_mask_full[:, :B] = pos_mask
    
    # InfoNCE: -log( sum_pos / sum_all )
    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * pos_mask_full).sum(dim=1)  # [B]
    all_sum = exp_sim.sum(dim=1)                     # [B]
    
    # Avoid log(0)
    eps = 1e-8
    loss = -torch.log((pos_sum + eps) / (all_sum + eps))
    
    # Only compute loss for anchors that have at least one positive
    has_pos = pos_sum > 0
    if has_pos.any():
        return loss[has_pos].mean()
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)
