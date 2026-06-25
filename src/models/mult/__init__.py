"""
MulT: Multimodal Transformer for Unaligned Sequences
Reference: Tsai et al., ACL 2019 (arXiv: 1906.00295)

Two variants:
  - MulT_Bimodal:   Audio + Video (legacy, for AVEC-style datasets)
  - MulT_AudioText: Audio + Text  (clinical, for DAIC-WOZ + hospital)
"""

from .mul_t_model import (
    # Audio-Video (legacy)
    MulT_Bimodal,
    # Audio-Text (clinical — primary use)
    MulT_AudioText,
    MultiScaleHead,
    build_scale_config_from_json,
    # Shared components
    CrossmodalAttention,
    CrossmodalTransformer,
    CrossmodalTransformerBlock,
    TemporalConv1D,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    EmotionLoss,
    ConcordanceCorrelationCoefficient,
)

__all__ = [
    # Models
    "MulT_Bimodal",
    "MulT_AudioText",
    "MultiScaleHead",
    "build_scale_config_from_json",
    # Core components
    "CrossmodalAttention",
    "CrossmodalTransformer",
    "CrossmodalTransformerBlock",
    "TemporalConv1D",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    # Losses
    "EmotionLoss",
    "ConcordanceCorrelationCoefficient",
]
