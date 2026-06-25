#!/usr/bin/env python3
"""
CancerEmotionSystem — Phase B: Multimodal Clinical Model
══════════════════════════════════════════════════════════
Audio + Text → Clinical Score Regression (PHQ-8/PHQ-9 + GAD-7 + PSQI)

Architecture:
  Audio:   wav2vec2-base(frozen) → TemporalConv → Transformer → ā
  Text:    BERT-base-uncased(frozen) → [CLS] → t̄
  Fusion:  [ā; t̄; ā⊙t̄; |ā-t̄|] → MLP → [PHQ, GAD, PSQI]
"""

import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np

def set_seed(seed=42):
    """Fix random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
from transformers import Wav2Vec2Model, BertModel, BertTokenizer


# ══════════════════════════════════════════════════════════
# 1. Audio Encoder (Phase A v1 architecture, feature-extraction mode)
# ══════════════════════════════════════════════════════════

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[: x.size(1), :])


class AudioEncoder(nn.Module):
    """
    Audio feature extractor: wav2vec2 → Conv → Transformer → pooled embedding.
    Output: [B, d_model] feature vector for regression.
    """
    def __init__(
        self,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        dropout=0.1,
        freeze_wav2vec2=True,
        pretrained_path=None,
    ):
        super().__init__()
        # Backbone
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        d_w2v = self.wav2vec2.config.hidden_size  # 768

        if freeze_wav2vec2:
            for p in self.wav2vec2.parameters():
                p.requires_grad = False

        # Projection: 768 → d_model
        self.input_proj = nn.Linear(d_w2v, d_model)

        # Temporal convolution (downsample)
        self.tconv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),
        )

        # Transformer
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=5000, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model

        # Load pretrained weights if provided
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            # Filter to audio encoder keys only
            audio_state = {
                k.replace("wav2vec2.", ""): v
                for k, v in state.items()
                if k.startswith(("wav2vec2.", "input_proj.", "tconv.", "transformer.", "pos_enc."))
            }
            if audio_state:
                missing, unexpected = self.load_state_dict(audio_state, strict=False)
                print(f"[AudioEncoder] Loaded pretrained: {len(audio_state)} params, {len(missing)} missing, {len(unexpected)} unexpected")

    def forward(self, waveform, attention_mask=None):
        """
        Args:
            waveform: [B, T_audio] raw 16kHz audio
            attention_mask: [B, T_audio] bool mask
        Returns:
            embedding: [B, d_model]
        """
        with torch.set_grad_enabled(
            not all(p.requires_grad == False for p in self.wav2vec2.parameters())
        ):
            w2v_out = self.wav2vec2(waveform, attention_mask=attention_mask)
            features = w2v_out.last_hidden_state  # [B, T_w2v, 768]

        features = self.input_proj(features)  # [B, T_w2v, d_model]
        features = features.transpose(1, 2)   # [B, d_model, T_w2v]
        features = self.tconv(features)       # [B, d_model, T_conv]
        features = features.transpose(1, 2)   # [B, T_conv, d_model]
        features = self.pos_enc(features)
        features = self.transformer(features) # [B, T_conv, d_model]
        pooled = features.mean(dim=1)         # [B, d_model]
        return pooled


# ══════════════════════════════════════════════════════════
# 2. Text Encoder
# ══════════════════════════════════════════════════════════

class TextEncoder(nn.Module):
    """
    Text encoder: BERT-base → [CLS] pooled embedding.
    """
    def __init__(self, d_model=256, freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        d_bert = self.bert.config.hidden_size  # 768

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(d_bert, d_model)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.d_model = d_model

    def forward(self, texts, device):
        """
        Args:
            texts: List[str] batch of transcript texts
            device: torch device
        Returns:
            embedding: [B, d_model]
        """
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.set_grad_enabled(
            not all(p.requires_grad == False for p in self.bert.parameters())
        ):
            outputs = self.bert(**tokens)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        return self.proj(cls_emb)  # [B, d_model]


# ══════════════════════════════════════════════════════════
# 3. Multimodal Fusion + Regression Head
# ══════════════════════════════════════════════════════════

class MultimodalFusion(nn.Module):
    """
    Fuse audio and text embeddings, predict clinical scores.
    """
    def __init__(
        self,
        d_model=256,
        n_tasks=3,          # PHQ, GAD, PSQI
        fusion_dropout=0.2,
        head_hidden=128,
    ):
        super().__init__()
        # Fusion: [ā; t̄; ā⊙t̄; |ā-t̄|]
        fusion_dim = d_model * 4  # 1024

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )

        # Multi-task regression heads
        self.heads = nn.ModuleDict({
            "phq": nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(head_hidden, 1),
            ),
            "gad": nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(head_hidden, 1),
            ),
            "psqi": nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(head_hidden, 1),
            ),
        })

    def forward(self, audio_emb, text_emb):
        """
        Args:
            audio_emb: [B, d_model]
            text_emb:  [B, d_model]
        Returns:
            dict of predictions: {"phq": [B,1], "gad": [B,1], "psqi": [B,1]}
        """
        # Fusion features
        fused = torch.cat([
            audio_emb,
            text_emb,
            audio_emb * text_emb,
            torch.abs(audio_emb - text_emb),
        ], dim=-1)  # [B, d_model*4]

        shared = self.fusion(fused)  # [B, d_model]

        # Multi-task predictions
        outputs = {}
        for task_name, head in self.heads.items():
            outputs[task_name] = head(shared)

        return outputs


# ══════════════════════════════════════════════════════════
# 4. Full Multimodal Clinical Model
# ══════════════════════════════════════════════════════════



# VRAM ESTIMATE
# =============
# wav2vec2-base (frozen):  ~1.2GB params + ~0.3GB activations @ batch=16
# bert-base-uncased (frozen): ~0.4GB params + ~0.2GB activations
# Fusion + heads (trainable): ~0.02GB params
# Total trainable: ~5.9M params (~24MB) — safe for any GPU >= 4GB
# Minimum recommended GPU: GTX 1080 (8GB) or better
# 2080Ti (11GB): comfortable, can increase batch to 32

class MultimodalClinicalModel(nn.Module):
    """
    End-to-end: Audio + Text → Clinical Scores.

    Usage:
        model = MultimodalClinicalModel(audio_pretrained="checkpoints/phaseA/phaseA_best.pt")
        audio = torch.randn(4, 160000)  # [B, 10s audio]
        texts = ["I feel tired all the time", "I can't sleep well", ...]
        preds = model(audio, texts)
        # preds = {"phq": tensor(...), "gad": tensor(...), "psqi": tensor(...)}
    """
    def __init__(
        self,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        dropout=0.1,
        freeze_audio_w2v=True,
        freeze_text_bert=True,
        n_tasks=3,
        audio_pretrained=None,
    ):
        super().__init__()
        self.d_model = d_model

        # Safety: verify freeze flags explicitly
        if not freeze_audio_w2v:
            print("\n*** WARNING: Unfreezing wav2vec2 (~320M params). VRAM usage will spike ***")
            print("*** Ensure >= 24GB GPU before proceeding ***\n")
        if not freeze_text_bert:
            print("\n*** WARNING: Unfreezing BERT (~110M params). VRAM usage will spike ***")
            print("*** Ensure >= 24GB GPU before proceeding ***\n")

        self.audio_encoder = AudioEncoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            freeze_wav2vec2=freeze_audio_w2v,
            pretrained_path=audio_pretrained,
        )

        self.text_encoder = TextEncoder(
            d_model=d_model,
            freeze_bert=freeze_text_bert,
        )

        self.fusion_head = MultimodalFusion(
            d_model=d_model,
            n_tasks=n_tasks,
            fusion_dropout=dropout,
        )

    def forward(self, waveform, texts, attention_mask=None):
        """
        Cross-modal fusion: audio and text are independently
        pooled to fixed [B, d_model] vectors, then concatenated.
        No temporal alignment needed — safe for any input length.

        Args:
            waveform: [B, T_audio] raw 16kHz audio
            texts: List[str] batch of transcripts
            attention_mask: optional [B, T_audio]
        Returns:
            dict: {"phq": [B,1], "gad": [B,1], "psqi": [B,1]}
        """
        device = waveform.device
        audio_emb = self.audio_encoder(waveform, attention_mask)  # [B, d_model]
        text_emb = self.text_encoder(texts, device)               # [B, d_model]
        return self.fusion_head(audio_emb, text_emb)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}



# ══════════════════════════════════════════════════════
# 5. Metrics
# ══════════════════════════════════════════════════════

def concordance_correlation_coefficient(y_pred, y_true):
    """
    Concordance Correlation Coefficient (CCC) — DAIC-WOZ official metric.
    
    Formula: CCC = 2 * cov(x,y) / (var(x) + var(y) + (mean(x) - mean(y))^2)
    
    Uses unbiased variance (ddof=1) for consistency with literature baselines.
    """
    y_pred = y_pred.view(-1).float()
    y_true = y_true.view(-1).float()
    
    mean_pred = y_pred.mean()
    mean_true = y_true.mean()
    
    # Center the variables
    x = y_pred - mean_pred
    y = y_true - mean_true
    
    # Covariance and variances (unbiased, ddof=1)
    n = y_pred.shape[0]
    cov = torch.sum(x * y) / (n - 1)
    var_pred = torch.sum(x * x) / (n - 1)
    var_true = torch.sum(y * y) / (n - 1)
    
    denom = var_pred + var_true + (mean_pred - mean_true) ** 2
    if denom < 1e-8:
        return torch.tensor(0.0, device=y_pred.device)
    
    ccc = 2 * cov / denom
    return torch.clamp(ccc, -1.0, 1.0)


def regression_metrics(y_pred, y_true):
    """Compute MAE, RMSE, Pearson r, and CCC."""
    y_pred = y_pred.view(-1).float()
    y_true = y_true.view(-1).float()
    
    mae = torch.abs(y_pred - y_true).mean()
    rmse = torch.sqrt(((y_pred - y_true) ** 2).mean())
    
    # Pearson correlation
    x = y_pred - y_pred.mean()
    y = y_true - y_true.mean()
    r_num = torch.sum(x * y)
    r_den = torch.sqrt(torch.sum(x * x) * torch.sum(y * y))
    r = r_num / (r_den + 1e-8)
    
    ccc = concordance_correlation_coefficient(y_pred, y_true)
    
    return {
        "mae": mae.item(),
        "rmse": rmse.item(),
        "pearson_r": r.item(),
        "ccc": ccc.item(),
    }


# ══════════════════════════════════════════════════════════
# 6. Quick Test
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═══ MultimodalClinicalModel — Architecture Test ═══\n")

    model = MultimodalClinicalModel(
        d_model=256,
        n_layers=4,
        n_heads=8,
        freeze_audio_w2v=True,
        freeze_text_bert=True,
        n_tasks=3,
        audio_pretrained=None,  # will load after v1 retraining
    )

    params = model.count_params()
    print(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Test forward pass
    print("\n[Test] Forward pass with random inputs...")
    B = 2
    T = 160000  # 10 seconds @ 16kHz
    dummy_audio = torch.randn(B, T)
    dummy_texts = ["I have been feeling down lately and cannot sleep well.",
                   "Everything is great, I feel energetic and happy."]

    with torch.no_grad():
        preds = model(dummy_audio, dummy_texts)

    for k, v in preds.items():
        print(f"  {k}: shape={v.shape}, mean={v.mean().item():.3f}")

    print("\n✅ Model built successfully")
    print(f"   Audio encoder dim: {model.d_model}")
    print(f"   Text encoder dim: {model.d_model}")
    print(f"   Fusion dim: {model.d_model * 4} → {model.d_model} (shared)")
    print(f"   Heads: {list(model.fusion_head.heads.keys())}")
