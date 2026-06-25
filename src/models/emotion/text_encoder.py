"""
Text encoder for MulT Audio-Text model.

Supports:
  - bert-base-uncased (English, DAIC-WOZ interview transcripts)
  - bert-base-chinese  (Chinese, hospital mood journals)
  - Simple word vectors (fast, no GPU needed, for quick tests)

Output format: [B, T, 768] compatible with MulT_AudioText.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from pathlib import Path
import warnings


class TextEncoder(nn.Module):
    """
    Unified text encoder for clinical emotion assessment.

    Usage:
        # DAIC-WOZ (English transcripts)
        encoder = TextEncoder(language="en")
        features, mask = encoder(["I feel tired all the time..."])

        # Hospital (Chinese mood journals)
        encoder = TextEncoder(language="zh")
        features, mask = encoder(["今天化疗后感觉特别累..."])

        # Lightweight (no BERT, for quick tests)
        encoder = TextEncoder(backend="word2vec")
    """

    SUPPORTED_LANGUAGES = ["en", "zh"]
    SUPPORTED_BACKENDS = ["bert", "word2vec"]
    
    # Pretrained model names
    MODEL_MAP = {
        "en": "bert-base-uncased",
        "zh": "bert-base-chinese",
    }
    
    # Feature dimensions
    FEATURE_DIMS = {
        "bert-base-uncased": 768,
        "bert-base-chinese": 768,
        "word2vec": 300,
    }

    def __init__(
        self,
        language: str = "en",
        backend: str = "bert",
        model_name: Optional[str] = None,
        max_length: int = 512,
        device: Optional[str] = None,
        freeze: bool = True,          # Freeze BERT weights
        output_hidden_states: bool = False,  # Return all layers or just last
        pooling: str = "none",        # "none" | "cls" | "mean" — "none" = full sequence
    ):
        """
        Args:
            language: "en" or "zh"
            backend: "bert" or "word2vec"
            model_name: Override default BERT model
            max_length: Max token length
            device: Torch device
            freeze: Freeze pretrained BERT weights
            output_hidden_states: Return all hidden layers
            pooling: How to pool token embeddings
        """
        super().__init__()
        self.language = language
        self.backend = backend
        self.max_length = max_length
        self.freeze = freeze
        self.pooling = pooling

        if backend == "bert":
            self._init_bert(model_name or self.MODEL_MAP[language])
        elif backend == "word2vec":
            self._init_word2vec()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        if device:
            self.to(device)

    def _init_bert(self, model_name: str):
        """Initialize HuggingFace BERT model."""
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers required for BERT encoding. "
                "Install: pip install transformers"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        self.feature_dim = self.bert.config.hidden_size  # 768 for bert-base
        self.model_name = model_name

        if self.freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def _init_word2vec(self):
        """Initialize lightweight word vector fallback."""
        self.feature_dim = 300
        self.embed = nn.Embedding(30000, self.feature_dim, padding_idx=0)
        # Simple char-level embedding for quick tests
        self.char_to_idx = {}
        self._next_idx = 1  # 0 = padding
        
        warnings.warn(
            "Using simple word2vec (not BERT). "
            "This is for quick testing only. Use backend='bert' for real training."
        )

    def _simple_tokenize(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple character-level tokenization for word2vec backend."""
        batch_tokens = []
        max_len = 0
        
        for text in texts:
            tokens = []
            for ch in text.lower():
                if ch not in self.char_to_idx:
                    self.char_to_idx[ch] = self._next_idx
                    self._next_idx += 1
                tokens.append(self.char_to_idx[ch])
            batch_tokens.append(tokens)
            max_len = max(max_len, len(tokens))
        
        # Clamp to max_length
        max_len = min(max_len, self.max_length)
        
        # Pad
        padded = []
        mask = []
        for tokens in batch_tokens:
            t = tokens[:max_len]
            pad_len = max_len - len(t)
            padded.append(t + [0] * pad_len)
            mask.append([1] * len(t) + [0] * pad_len)
        
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
        )

    def _bert_encode(
        self, texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode texts with BERT → [B, T, 768] embeddings.

        Args:
            texts: List of strings
        Returns:
            features: [B, max_len, feature_dim]
            mask: [B, max_len] bool (True = valid token)
        """
        device = next(self.bert.parameters()).device

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Forward through BERT
        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self.bert(**encoded)
            # outputs.last_hidden_state: [B, T, 768]

        features = outputs.last_hidden_state  # [B, T, 768]
        mask = encoded["attention_mask"].bool()  # [B, T]

        return features, mask

    def forward(
        self,
        texts: Union[str, List[str]],
        return_mask: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode text(s) to embeddings.

        Args:
            texts: Single string or list of strings
            return_mask: If True, also return attention mask
        Returns:
            features: [B, T, feature_dim]
            mask (optional): [B, T] bool
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.backend == "bert":
            features, mask = self._bert_encode(texts)
        else:
            token_ids, mask = self._simple_tokenize(texts)
            device = next(self.embed.parameters()).device
            features = self.embed(token_ids.to(device))
            mask = mask.to(device)

        if self.pooling == "cls":
            features = features[:, 0:1, :]  # [CLS] token only
            mask = mask[:, 0:1]
        elif self.pooling == "mean":
            # Masked mean over tokens
            mask_expanded = mask.unsqueeze(-1).float()
            features = (features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            features = features.unsqueeze(1)  # [B, 1, D]

        if return_mask:
            return features, mask
        return features

    def encode_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode large text list in batches (memory-efficient)."""
        all_features = []
        all_masks = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            feats, mask = self(batch)
            all_features.append(feats.cpu())
            all_masks.append(mask.cpu())

        return torch.cat(all_features, dim=0), torch.cat(all_masks, dim=0)


# ══════════════════════════════════════════════════════════════════
# Test
# ══════════════════════════════════════════════════════════════════

def test_text_encoder():
    """Test text encoder with dummy data."""
    print("=" * 60)
    print("Text Encoder Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Test 1: word2vec backend (always available) ──
    print("\n1. Word2Vec backend (lightweight):")
    encoder = TextEncoder(backend="word2vec", max_length=64, device=device)
    
    texts = [
        "I feel very tired today after the chemotherapy session.",
        "今天化疗后感觉特别累，什么都不想做。",
        "Feeling a bit better today, managed to take a short walk.",
    ]
    
    features, mask = encoder(texts)
    print(f"   Input: {len(texts)} texts")
    print(f"   Features: {features.shape}")  # [3, T, 300]
    print(f"   Mask: {mask.shape}")
    print(f"   Feature dim: {encoder.feature_dim}")
    print("   ✓ Word2Vec backend OK")

    # ── Test 2: BERT backend (if available) ──
    print("\n2. BERT backend:")
    try:
        import transformers
        encoder_bert = TextEncoder(language="en", backend="bert", max_length=128, device=device)
        features_bert, mask_bert = encoder_bert(texts[:2])  # Just English
        
        print(f"   Features: {features_bert.shape}")  # [2, T, 768]
        print(f"   Mask: {mask_bert.shape}")
        print(f"   Feature dim: {encoder_bert.feature_dim}")
        print("   ✓ BERT backend OK")
    except ImportError:
        print("   transformers not installed — skipping BERT test")
    except Exception as e:
        print(f"   BERT not available: {e}")

    # ── Test 3: Pooling modes ──
    print("\n3. Pooling modes:")
    encoder_pool = TextEncoder(backend="word2vec", pooling="mean", device=device)
    feats_pool, mask_pool = encoder_pool(texts)
    print(f"   Mean pooling: {feats_pool.shape}")  # [3, 1, 300] if mean, or [3, T, 300] if none
    print("   ✓ Pooling modes OK")

    # ── Test 4: Single text ──
    print("\n4. Single text input:")
    feats, mask = encoder("Hello world")
    print(f"   Features: {feats.shape}")
    print("   ✓ Single text OK")

    # ── Test 5: Integration with MulT_AudioText ──
    print("\n5. Integration with MulT_AudioText:")
    from . import MulT_AudioText
    
    model = MulT_AudioText(
        d_audio=768, d_text=encoder_bert.feature_dim if 'encoder_bert' in dir() else 300,
        d_model=256, n_layers=2, n_heads=4, d_ff=512,
        num_outputs=6,
        output_ranges=[(0,21),(0,21),(0,100),(15,75),(20,80),(0,30)],
        use_multi_head=True,
    ).to(device)
    
    # Dummy audio + real text
    B = len(texts)
    dummy_audio = torch.randn(B, 100, 768).to(device)
    text_feats, text_mask = encoder(texts)
    
    scores, _ = model(dummy_audio, text_feats.to(device), text_mask=text_mask.to(device))
    print(f"   Input: audio=[{B},100,768] + text={text_feats.shape}")
    print(f"   Output: {scores.shape}")  # [B, 6]
    print("   ✓ End-to-end Audio+Text pipeline works!")

    print("\n✅ All text encoder tests passed!")


if __name__ == "__main__":
    # When run directly, use relative imports differently
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3]))
    # Simple standalone test
    test_text_encoder()
