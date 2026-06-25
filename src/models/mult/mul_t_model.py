#!/usr/bin/env python3
"""
MulT — Multimodal Transformer for Unaligned Sequences
Reference: Tsai et al., "Multimodal Transformer for Unaligned Multimodal
           Language Sequences", ACL 2019. (arXiv: 1906.00295)

This is the core cross-modal attention module that enables temporal alignment
between modalities with different sampling rates without explicit synchronization.

Architecture (Audio-Video):
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  Audio   │    │  Video   │    │  Text    │
    │ Encoder  │    │ Encoder  │    │ Encoder  │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
    ┌────▼─────┐    ┌────▼─────┐    ┌────▼─────┐
    │ Temporal │    │ Temporal │    │ Temporal │
    │ Conv1D   │    │ Conv1D   │    │ Conv1D   │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         └───────┬───────┴───────┬───────┘
                 │               │
         ┌───────▼───────┐ ┌─────▼─────────┐
         │  Crossmodal   │ │  Crossmodal    │
         │  Transformer  │ │  Transformer   │
         │  (V→A, T→A)   │ │  (A→V, T→V)   │
         └───────────────┘ └───────────────┘
                 │               │
                 └───────┬───────┘
                         │
                 ┌───────▼───────┐
                 │   Fusion &    │
                 │   Prediction  │
                 └───────────────┘

Architecture (Audio-Text, for clinical use):
    ┌──────────┐         ┌──────────┐
    │  Audio   │         │  Text    │
    │ Encoder  │         │ Encoder  │
    └────┬─────┘         └────┬─────┘
         │                    │
    ┌────▼─────┐         ┌────▼─────┐
    │ Temporal │         │ Temporal │
    │ Conv1D   │         │ Conv1D   │
    └────┬─────┘         └────┬─────┘
         │                    │
         └────────┬───────────┘
                  │
         ┌────────▼────────┐
         │  Crossmodal     │
         │  Transformers   │
         │  (A→T, T→A)     │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │   Fusion &      │
         │   Prediction    │
         └─────────────────┘
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# Positional Encoding
# ══════════════════════════════════════════════════════════════════

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in "Attention Is All You Need".

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + positional encoding: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.max_len}"
            )
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding (alternative to sinusoidal)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) / math.sqrt(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════
# Temporal Convolution (for local temporal structure)
# ══════════════════════════════════════════════════════════════════

class TemporalConv1D(nn.Module):
    """
    1D convolution over the time dimension to capture local temporal context
    before feeding into cross-modal attention.
    """

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # same-length output
            groups=d_model,  # depthwise
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        residual = x
        # Conv1D expects [B, C, T]
        x = x.transpose(1, 2)  # [B, d_model, T]
        x = self.conv(x)       # [B, d_model, T]
        x = x.transpose(1, 2)  # [B, T, d_model]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x


# ══════════════════════════════════════════════════════════════════
# Crossmodal Attention (THE CORE OF MulT)
# ══════════════════════════════════════════════════════════════════

class CrossmodalAttention(nn.Module):
    """
    Crossmodal Attention: one modality's Query attends to another modality's Key/Value.

    This is the key innovation of MulT:
    - Q comes from modality α (e.g., Audio)
    - K and V come from modality β (e.g., Video)
    - This allows the model to learn cross-modal temporal correspondences
      WITHOUT requiring the modalities to be aligned at the same sampling rate.

    Shape convention:
        Q: [batch, seq_len_alpha, d_model]
        K: [batch, seq_len_beta,  d_model]
        V: [batch, seq_len_beta,  d_model]
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d_model] → [B, n_heads, T, d_k]"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, n_heads, T, d_k] → [B, T, d_model]"""
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(
        self,
        x_q: torch.Tensor,  # Query modality: [B, T_q, d_model]
        x_kv: torch.Tensor,  # Key/Value modality: [B, T_kv, d_model]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-modal attention: Q from modality α attends to K,V from modality β.

        Args:
            x_q:  Query sequence from modality α  [B, T_q, d_model]
            x_kv: Key/Value sequence from modality β [B, T_kv, d_model]
            mask: Optional attention mask [B, T_q, T_kv]

        Returns:
            Context-augmented representation for modality α: [B, T_q, d_model]
        """
        residual = x_q

        # Linear projections + split heads
        Q = self._split_heads(self.W_q(x_q))    # [B, H, T_q, d_k]
        K = self._split_heads(self.W_k(x_kv))   # [B, H, T_kv, d_k]
        V = self._split_heads(self.W_v(x_kv))   # [B, H, T_kv, d_k]

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, T_q, T_kv]

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, H, T_q, T_kv]
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)

        # Weighted sum of values
        context = torch.matmul(attn_weights, V)  # [B, H, T_q, d_k]
        context = self._merge_heads(context)     # [B, T_q, d_model]

        # Output projection + residual
        output = self.W_o(context)
        output = self.dropout_out(output)
        return output + residual  # residual connection


# ══════════════════════════════════════════════════════════════════
# Crossmodal Transformer Block
# ══════════════════════════════════════════════════════════════════

class CrossmodalTransformerBlock(nn.Module):
    """
    A single Crossmodal Transformer block.

    For modality α attending to modality β:
        x_α ← CrossmodalAttention(Q=x_α, KV=x_β)
        x_α ← FFN(x_α)
    """

    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = CrossmodalAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_q:  Query modality  [B, T_q, d_model]
            x_kv: Key/Value modality [B, T_kv, d_model]
        Returns:
            Updated query modality [B, T_q, d_model]
        """
        x_q = self.norm1(self.cross_attn(x_q, x_kv))
        x_q = self.norm2(x_q + self.ffn(x_q))
        return x_q


class CrossmodalTransformer(nn.Module):
    """
    Stack of Crossmodal Transformer blocks.

    In MulT, each target modality has D layers of crossmodal attention
    attending to each source modality.
    """

    def __init__(self, d_model: int, n_layers: int = 4, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossmodalTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x_q = layer(x_q, x_kv)
        return x_q


# ══════════════════════════════════════════════════════════════════
# Full MulT Model (Bi-modal: Audio + Video)
# ══════════════════════════════════════════════════════════════════

class MulT_Bimodal(nn.Module):
    """
    MulT adapted for Audio-Visual emotion recognition (bi-modal version).

    For cancer patient emotion assessment:
      - Modality A: Audio (voice features from wav2vec2 / MFCC)
      - Modality V: Video (facial features from CNN / ViT)

    Architecture flow:
        1. Encode each modality independently
        2. Apply temporal convolution for local context
        3. Add positional encoding
        4. Crossmodal Transformers:
           - Audio ← Video (V attends to A's Key/Value)
           - Video ← Audio (A attends to V's Key/Value)
        5. Concatenate + predict emotion score
    """

    def __init__(
        self,
        d_audio: int = 768,     # Audio feature dimension (e.g., wav2vec2 output)
        d_video: int = 512,     # Video feature dimension (e.g., ViT output)
        d_model: int = 256,     # Common embedding dimension
        n_layers: int = 4,      # Crossmodal Transformer layers
        n_heads: int = 8,       # Attention heads
        d_ff: int = 1024,       # FFN hidden dimension
        dropout: float = 0.1,
        max_seq_len: int = 500, # Max sequence length (seconds * fps)
        num_emotions: int = 2,  # Number of output scores (Valence, Arousal or Depression, Anxiety)
    ):
        super().__init__()
        self.d_model = d_model
        self.num_emotions = num_emotions

        # ── 1. Modality-specific projection layers ──
        self.audio_proj = nn.Sequential(
            nn.Linear(d_audio, d_model),
            nn.LayerNorm(d_model),
        )
        self.video_proj = nn.Sequential(
            nn.Linear(d_video, d_model),
            nn.LayerNorm(d_model),
        )

        # ── 2. Temporal convolutions ──
        self.audio_tconv = TemporalConv1D(d_model, kernel_size=3, dropout=dropout)
        self.video_tconv = TemporalConv1D(d_model, kernel_size=3, dropout=dropout)

        # ── 3. Positional encoding ──
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # ── 4. Crossmodal Transformers ──
        # Audio attending to Video
        self.audio_from_video = CrossmodalTransformer(
            d_model, n_layers, n_heads, d_ff, dropout
        )
        # Video attending to Audio
        self.video_from_audio = CrossmodalTransformer(
            d_model, n_layers, n_heads, d_ff, dropout
        )

        # ── 5. Self-attention after cross-modal (optional) ──
        self.audio_self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.video_self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # ── 6. Fusion + Prediction head ──
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Mean pooling + prediction
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_emotions),
        )

        # For regression: output raw scores
        # For classification: add sigmoid

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        audio_features: torch.Tensor,   # [B, T_a, d_audio]
        video_features: torch.Tensor,   # [B, T_v, d_video]
        audio_mask: Optional[torch.Tensor] = None,   # [B, T_a]
        video_mask: Optional[torch.Tensor] = None,   # [B, T_v]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            audio_features: Pre-extracted audio features [B, T_a, d_audio]
            video_features: Pre-extracted video features [B, T_v, d_video]
            audio_mask: Boolean mask for valid audio timesteps [B, T_a]
            video_mask: Boolean mask for valid video timesteps [B, T_v]

        Returns:
            emotion_scores: [B, num_emotions] — Valence/Arousal or Depression/Anxiety
            aux_outputs: dict with intermediate representations for analysis
        """
        B = audio_features.size(0)

        # ── 1. Project to common dimension ──
        audio_emb = self.audio_proj(audio_features)  # [B, T_a, d_model]
        video_emb = self.video_proj(video_features)  # [B, T_v, d_model]

        # ── 2. Temporal convolution ──
        audio_emb = self.audio_tconv(audio_emb)
        video_emb = self.video_tconv(video_emb)

        # ── 3. Positional encoding ──
        audio_emb = self.pos_enc(audio_emb)
        video_emb = self.pos_enc(video_emb)

        # ── 4. Crossmodal Transformers ──
        # Audio enriched by Video context
        audio_cross = self.audio_from_video(audio_emb, video_emb)  # [B, T_a, d_model]
        # Video enriched by Audio context
        video_cross = self.video_from_audio(video_emb, audio_emb)  # [B, T_v, d_model]

        # ── 5. Self-attention refinement ──
        audio_cross, _ = self.audio_self_attn(audio_cross, audio_cross, audio_cross)
        video_cross, _ = self.video_self_attn(video_cross, video_cross, video_cross)

        # ── 6. Temporal pooling → global representation ──
        if audio_mask is not None:
            audio_mask_expanded = audio_mask.unsqueeze(-1).float()  # [B, T_a, 1]
            audio_global = (audio_cross * audio_mask_expanded).sum(dim=1) / audio_mask_expanded.sum(dim=1).clamp(min=1)
        else:
            audio_global = audio_cross.mean(dim=1)  # [B, d_model]

        if video_mask is not None:
            video_mask_expanded = video_mask.unsqueeze(-1).float()
            video_global = (video_cross * video_mask_expanded).sum(dim=1) / video_mask_expanded.sum(dim=1).clamp(min=1)
        else:
            video_global = video_cross.mean(dim=1)  # [B, d_model]

        # ── 7. Concatenate + predict ──
        fused = self.fusion_proj(torch.cat([audio_global, video_global], dim=-1))  # [B, d_model]
        emotion_scores = self.pred_head(fused)  # [B, num_emotions]

        # Auxiliary outputs for interpretability
        aux = {
            "audio_cross": audio_cross,
            "video_cross": video_cross,
            "audio_global": audio_global,
            "video_global": video_global,
            "fused": fused,
        }

        return emotion_scores, aux


# ══════════════════════════════════════════════════════════════════
# MulT Audio-Text: Clinical Bi-modal Version
# ══════════════════════════════════════════════════════════════════

class MulT_AudioText(nn.Module):
    """
    MulT adapted for Audio-Text bimodal clinical emotion assessment.

    This is the primary model for the cancer patient emotion system:
      - Modality A: Audio (voice from wav2vec2 / MFCC / HuBERT)
      - Modality T: Text  (transcripts or mood journals from BERT)

    Designed for:
      Phase A (DAIC-WOZ):  [Audio + Interview Transcript] → PHQ-8
      Phase C (Hospital):  [Audio + Mood Journal Text] → Custom Scales

    Architecture:
        1. Encode each modality independently
        2. Temporal convolution for local context
        3. + Positional encoding
        4. Crossmodal Transformers:
           Audio ← Text  (audio enriched by text context)
           Text  ← Audio (text enriched by audio context)
        5. Mean pool → Concatenate → Multi-task prediction heads
    """

    def __init__(
        self,
        d_audio: int = 768,      # Audio feature dim (wav2vec2: 768, HuBERT: 1024)
        d_text: int = 768,        # Text feature dim (BERT: 768)
        d_model: int = 256,       # Common embedding dimension
        n_layers: int = 4,        # Crossmodal Transformer layers per direction
        n_heads: int = 8,         # Attention heads
        d_ff: int = 1024,         # FFN hidden dim
        dropout: float = 0.1,
        max_seq_len: int = 500,
        num_outputs: int = 1,     # Number of output scores (e.g., 1 for PHQ-8, 6 for multi-scale)
        output_ranges: list = None,  # [(min, max), ...] per output, for scaled sigmoid heads
        use_multi_head: bool = False,  # True → MultiScaleHead, False → simple MLP
    ):
        super().__init__()
        self.d_model = d_model
        self.num_outputs = num_outputs
        self.use_multi_head = use_multi_head

        # ── 1. Modality-specific projection ──
        self.audio_proj = nn.Sequential(
            nn.Linear(d_audio, d_model),
            nn.LayerNorm(d_model),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_model),
            nn.LayerNorm(d_model),
        )

        # ── 2. Temporal convolutions ──
        self.audio_tconv = TemporalConv1D(d_model, kernel_size=3, dropout=dropout)
        self.text_tconv = TemporalConv1D(d_model, kernel_size=3, dropout=dropout)

        # ── 3. Positional encoding ──
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # ── 4. Crossmodal Transformers (bi-directional) ──
        self.audio_from_text = CrossmodalTransformer(d_model, n_layers, n_heads, d_ff, dropout)
        self.text_from_audio = CrossmodalTransformer(d_model, n_layers, n_heads, d_ff, dropout)

        # ── 5. Self-attention refinement ──
        self.audio_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.text_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # ── 6. Fusion ──
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 7. Prediction head(s) ──
        if use_multi_head and output_ranges:
            self.pred_head = MultiScaleHead(d_model, output_ranges, dropout)
        else:
            self.pred_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_outputs),
            )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        audio_features: torch.Tensor,       # [B, T_a, d_audio]
        text_features: torch.Tensor,         # [B, T_t, d_text]
        audio_mask: Optional[torch.Tensor] = None,   # [B, T_a] bool
        text_mask: Optional[torch.Tensor] = None,    # [B, T_t] bool
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            audio_features: Pre-extracted audio embeddings
            text_features:  Pre-extracted text embeddings (BERT/word vectors)
            audio_mask:     Valid-frame mask for audio
            text_mask:      Valid-token mask for text
            return_aux:     If True, return intermediate representations

        Returns:
            predictions: [B, num_outputs] clinical scale scores
            aux: dict of intermediate representations
        """
        # ── 1. Project to common dim ──
        audio_emb = self.audio_proj(audio_features)  # [B, T_a, d_model]
        text_emb = self.text_proj(text_features)      # [B, T_t, d_model]

        # ── 2. Temporal convolution ──
        audio_emb = self.audio_tconv(audio_emb)
        text_emb = self.text_tconv(text_emb)

        # ── 3. Positional encoding ──
        audio_emb = self.pos_enc(audio_emb)
        text_emb = self.pos_enc(text_emb)

        # ── 4. Crossmodal Transformers ──
        # Audio enriched by text context ("what was said affects how it sounds")
        audio_cross = self.audio_from_text(audio_emb, text_emb)
        # Text enriched by audio context ("how it was said affects what was said")
        text_cross = self.text_from_audio(text_emb, audio_emb)

        # ── 5. Self-attention ──
        audio_cross, _ = self.audio_self_attn(audio_cross, audio_cross, audio_cross)
        text_cross, _ = self.text_self_attn(text_cross, text_cross, text_cross)

        # ── 6. Masked mean pooling → global ──
        if audio_mask is not None:
            a_mask = audio_mask.unsqueeze(-1).float()
            audio_global = (audio_cross * a_mask).sum(dim=1) / a_mask.sum(dim=1).clamp(min=1)
        else:
            audio_global = audio_cross.mean(dim=1)

        if text_mask is not None:
            t_mask = text_mask.unsqueeze(-1).float()
            text_global = (text_cross * t_mask).sum(dim=1) / t_mask.sum(dim=1).clamp(min=1)
        else:
            text_global = text_cross.mean(dim=1)

        # ── 7. Fusion + predict ──
        fused = self.fusion_proj(torch.cat([audio_global, text_global], dim=-1))
        predictions = self.pred_head(fused)

        aux = {}
        if return_aux:
            aux = {
                "audio_cross": audio_cross,
                "text_cross": text_cross,
                "audio_global": audio_global,
                "text_global": text_global,
                "fused": fused,
            }

        return predictions, aux

    def freeze_encoder(self):
        """Freeze all layers except prediction head (for transfer learning)."""
        for name, param in self.named_parameters():
            if not name.startswith("pred_head"):
                param.requires_grad = False

    def unfreeze_encoder(self, lr_factor: float = 0.1):
        """Unfreeze encoder with lower learning rate for fine-tuning."""
        for name, param in self.named_parameters():
            param.requires_grad = True
        # Return param groups for optimizer
        encoder_params = [p for n, p in self.named_parameters() if not n.startswith("pred_head")]
        head_params = [p for n, p in self.named_parameters() if n.startswith("pred_head")]
        return [
            {"params": encoder_params, "lr_factor": lr_factor},
            {"params": head_params, "lr_factor": 1.0},
        ]


# ══════════════════════════════════════════════════════════════════
# Multi-Scale Prediction Head (for hospital clinical scales)
# ══════════════════════════════════════════════════════════════════

class MultiScaleHead(nn.Module):
    """
    Multi-task prediction head: one output per clinical scale.

    Each scale gets its own small MLP with sigmoid output scaled to its range.

    Supported scales (defined in configs/scales/):
      HADS-A  (anxiety):         0–21
      HADS-D  (depression):      0–21
      VAS     (pain):            0–100
      CFS     (cancer fatigue):  15–75
      PROMIS  (physical func):   T-score ~20–80
      LSNS-6  (social network):  0–30
    """

    def __init__(
        self,
        d_model: int = 256,
        output_ranges: list = None,  # [(min, max), ...] one per output
        dropout: float = 0.1,
        scale_names: list = None,    # Optional: ["hads_a", "hads_d", ...]
    ):
        super().__init__()
        if output_ranges is None:
            output_ranges = [(0, 1)]  # Default: single unit-range output

        self.num_scales = len(output_ranges)
        self.output_ranges = output_ranges
        self.scale_names = scale_names or [f"scale_{i}" for i in range(self.num_scales)]

        # One small MLP per scale
        self.heads = nn.ModuleDict()
        for i, (lo, hi) in enumerate(output_ranges):
            name = self.scale_names[i]
            self.heads[name] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),  # outputs in [0, 1]
            )
            # Store range for scaling
            self.heads[name].scale_lo = lo
            self.heads[name].scale_hi = hi

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_embedding: [B, d_model]
        Returns:
            predictions: [B, num_scales] — each column in its scale's native range
        """
        outputs = []
        for name in self.scale_names:
            head = self.heads[name]
            raw = head(fused_embedding)  # [B, 1] in [0, 1]
            scaled = raw * (head.scale_hi - head.scale_lo) + head.scale_lo  # [B, 1]
            outputs.append(scaled)
        return torch.cat(outputs, dim=-1)  # [B, num_scales]

    def predict_individual(self, fused_embedding, scale_name):
        """Predict a single scale given its name."""
        head = self.heads[scale_name]
        raw = head(fused_embedding)
        return raw * (head.scale_hi - head.scale_lo) + head.scale_lo


# ══════════════════════════════════════════════════════════════════
# Hospital Scale Configuration Builder
# ══════════════════════════════════════════════════════════════════

def build_scale_config_from_json(scales_dir: str = None) -> Tuple[list, list]:
    """
    Read scale JSON files and build output_ranges + scale_names for MultiScaleHead.

    Args:
        scales_dir: Path to configs/scales/ directory
    Returns:
        output_ranges: [(min, max), ...]
        scale_names:   ["hads_anxiety", "hads_depression", ...]
    """
    import json
    from pathlib import Path

    if scales_dir is None:
        scales_dir = Path(__file__).parents[3] / "configs" / "scales"

    scales_dir = Path(scales_dir)

    # Mapping: JSON filename → output name + range extraction
    scale_configs = []

    if (scales_dir / "HADS.json").exists():
        scale_configs.extend([
            ("hads_anxiety", 0, 21),
            ("hads_depression", 0, 21),
        ])
    if (scales_dir / "VAS.json").exists():
        scale_configs.append(("vas_pain", 0, 100))
    if (scales_dir / "CFS.json").exists():
        scale_configs.append(("cfs_total", 15, 75))
    if (scales_dir / "PROMIS_Physical_Function_6b.json").exists():
        scale_configs.append(("promis_pf_t", 20, 80))
    if (scales_dir / "LSNS-6.json").exists():
        scale_configs.append(("lsns6_total", 0, 30))

    scale_names = [s[0] for s in scale_configs]
    output_ranges = [(s[1], s[2]) for s in scale_configs]

    return output_ranges, scale_names

class ConcordanceCorrelationCoefficient(nn.Module):
    """
    Concordance Correlation Coefficient (CCC) loss.
    Commonly used in affective computing for continuous emotion prediction.

    CCC measures agreement between two variables:
        CCC = 2 * ρ * σ_x * σ_y / (σ_x² + σ_y² + (μ_x - μ_y)²)

    where ρ is Pearson correlation.
    Loss = 1 - CCC  (to minimize)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: [B, D] predicted scores
            y_true: [B, D] ground truth scores
        Returns:
            CCC loss per dimension, averaged
        """
        B = y_true.size(0)
        if B < 2:
            return torch.tensor(0.0, device=y_pred.device)

        mean_pred = y_pred.mean(dim=0, keepdim=True)
        mean_true = y_true.mean(dim=0, keepdim=True)

        var_pred = y_pred.var(dim=0, unbiased=False) + self.eps
        var_true = y_true.var(dim=0, unbiased=False) + self.eps

        # Pearson correlation
        cov = ((y_pred - mean_pred) * (y_true - mean_true)).mean(dim=0)

        # CCC
        numerator = 2 * cov
        denominator = var_pred + var_true + (mean_pred - mean_true).pow(2).squeeze(0) + self.eps
        ccc = numerator / denominator

        return 1.0 - ccc.mean()


class EmotionLoss(nn.Module):
    """
    Combined loss for emotion regression:
        Total = MSE + α * CCC_Loss

    MSE  → penalizes absolute error
    CCC  → encourages correlation (trend accuracy)
    """

    def __init__(self, ccc_weight: float = 0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ccc = ConcordanceCorrelationCoefficient()
        self.ccc_weight = ccc_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        mse_loss = self.mse(y_pred, y_true)
        ccc_loss = self.ccc(y_pred, y_true)
        total = mse_loss + self.ccc_weight * ccc_loss

        losses = {
            "total": total.item(),
            "mse": mse_loss.item(),
            "ccc": ccc_loss.item(),
        }
        return total, losses


# ══════════════════════════════════════════════════════════════════
# Test / Sanity Check
# ══════════════════════════════════════════════════════════════════

def test_mul_t_audiovideo():
    """Sanity check: MulT Audio-Video (legacy)."""
    print("=" * 60)
    print("MulT Audio-Video Sanity Check")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    batch_size = 4
    T_audio, T_video = 100, 75
    d_audio, d_video = 768, 512

    model = MulT_Bimodal(
        d_audio=d_audio, d_video=d_video,
        d_model=256, n_layers=2, n_heads=4, d_ff=512,
        dropout=0.1, num_emotions=2,
    ).to(device)

    audio = torch.randn(batch_size, T_audio, d_audio).to(device)
    video = torch.randn(batch_size, T_video, d_video).to(device)
    scores, aux = model(audio, video)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Input:  Audio={audio.shape}, Video={video.shape}")
    print(f"  Output: {scores.shape}")
    print(f"  Params: {n_params:,}")
    print("  ✓ Audio-Video MulT OK\n")
    return True


def test_mul_t_audiotext():
    """Sanity check: MulT Audio-Text (clinical version)."""
    print("=" * 60)
    print("MulT Audio-Text (Clinical) Sanity Check")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    batch_size = 4
    T_audio = 100   # ~3s audio @ 33Hz frame rate
    T_text = 50     # ~50 tokens (short mood journal)
    d_audio = 768   # wav2vec2
    d_text = 768    # BERT

    # ── Test 1: Single output (DAIC-WOZ PHQ-8 mode) ──
    print("\n1. Single-output mode (PHQ-8, DAIC-WOZ pretraining):")
    model = MulT_AudioText(
        d_audio=d_audio, d_text=d_text,
        d_model=256, n_layers=2, n_heads=4, d_ff=512,
        dropout=0.1, num_outputs=1,
    ).to(device)

    audio = torch.randn(batch_size, T_audio, d_audio).to(device)
    text = torch.randn(batch_size, T_text, d_text).to(device)
    scores, aux = model(audio, text)

    print(f"  Audio: {audio.shape}")
    print(f"  Text:  {text.shape}")
    print(f"  Pred:  {scores.shape}")  # [4, 1]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Verify gradients
    criterion = nn.MSELoss()
    loss = criterion(scores, torch.randn_like(scores))
    loss.backward()
    print(f"  Loss:  {loss.item():.4f}")
    print("  ✓ Single-output mode OK")

    # ── Test 2: Multi-scale mode (hospital 6 scales) ──
    print("\n2. Multi-scale mode (HADS-A, HADS-D, VAS, CFS, PROMIS, LSNS-6):")
    output_ranges = [
        (0, 21),    # hads_anxiety
        (0, 21),    # hads_depression
        (0, 100),   # vas_pain
        (15, 75),   # cfs_total
        (20, 80),   # promis_pf_t
        (0, 30),    # lsns6_total
    ]
    scale_names = [
        "hads_anxiety", "hads_depression", "vas_pain",
        "cfs_total", "promis_pf_t", "lsns6_total",
    ]

    model_ms = MulT_AudioText(
        d_audio=d_audio, d_text=d_text,
        d_model=256, n_layers=2, n_heads=4, d_ff=512,
        dropout=0.1, num_outputs=6,
        output_ranges=output_ranges,
        use_multi_head=True,
    ).to(device)

    scores_ms, aux_ms = model_ms(audio, text)

    print(f"  Pred:  {scores_ms.shape}")  # [4, 6]
    print(f"  Scales: {scale_names}")
    print(f"  Sample predictions (first batch):")
    for i, name in enumerate(scale_names):
        lo, hi = output_ranges[i]
        val = scores_ms[0, i].item()
        print(f"    {name:16s}: {val:6.2f}  (range: {lo}-{hi})")

    # Verify all scores within expected ranges
    for i, (lo, hi) in enumerate(output_ranges):
        assert scores_ms[:, i].min() >= lo - 0.1, f"{scale_names[i]} below range"
        assert scores_ms[:, i].max() <= hi + 0.1, f"{scale_names[i]} above range"
    print("  ✓ All scores within expected ranges")

    n_params_ms = sum(p.numel() for p in model_ms.parameters())
    print(f"  Params: {n_params_ms:,}")

    # ── Test 3: freeze_encoder → transfer learning simulation ──
    print("\n3. Transfer learning (freeze encoder → only train heads):")
    model_ms.freeze_encoder()
    frozen = sum(not p.requires_grad for p in model_ms.parameters())
    trainable = sum(p.requires_grad for p in model_ms.parameters())
    print(f"  Frozen params:    {frozen:,}")
    print(f"  Trainable params: {trainable:,}")
    assert frozen > trainable, "Encoder should have more params than heads"
    print("  ✓ freeze_encoder() works correctly")

    # ── Test 4: Unaligned sequence lengths ──
    print("\n4. Unaligned modalities (different T):")
    audio_short = torch.randn(batch_size, 80, d_audio).to(device)   # shorter audio
    text_long = torch.randn(batch_size, 120, d_text).to(device)     # longer text
    scores3, _ = model_ms(audio_short, text_long)
    print(f"  Audio(80) + Text(120) → {scores3.shape}  ✓")

    # ── Test 5: build_scale_config_from_json ──
    print("\n5. Scale config builder:")
    ranges, names = build_scale_config_from_json()
    print(f"  Scales found: {len(names)}")
    for n, (lo, hi) in zip(names, ranges):
        print(f"    {n}: [{lo}, {hi}]")
    assert len(names) == 6, f"Expected 6 scales, got {len(names)}"
    print("  ✓ Scale config builder OK")

    print("\n✅ All MulT Audio-Text sanity checks passed!")
    return model_ms


if __name__ == "__main__":
    test_mul_t_audiovideo()
    test_mul_t_audiotext()
