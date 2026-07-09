#!/usr/bin/env python3
"""
WavLM Multi-Layer Adapter for Depression Detection.
Frozen WavLM (layers 8-12) + Frozen RoBERTa + OpenFace → Cross-Attention → Classifier.
"""
import torch, torch.nn as nn
import numpy as np

class AudioAdapter(nn.Module):
    """WavLM frozen, multi-layer features (8-12) → concat → adapter → embedding"""
    def __init__(self, d_model=256, bottleneck=256, dropout=0.2):
        super().__init__()
        from transformers import WavLMModel
        self.wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base-plus')
        for p in self.wavlm.parameters(): p.requires_grad = False
        d_cat = 768 * 5  # layers 8,9,10,11,12
        self.adapter = nn.Sequential(
            nn.Linear(d_cat, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, waveform, attention_mask=None):
        with torch.no_grad():
            out = self.wavlm(waveform, attention_mask=attention_mask, output_hidden_states=True)
            hs = out.hidden_states  # tuple of 13 layers (0-12)
            # Concat layers 8,9,10,11,12 (0-indexed)
            cat = torch.cat([hs[i] for i in [8,9,10,11,12]], dim=-1)
        if attention_mask is not None:
            ds = max(1, attention_mask.shape[1] // cat.shape[1])
            m = attention_mask[:, ::ds][:, :cat.shape[1]].unsqueeze(-1).float()
            pooled = (cat * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            pooled = cat.mean(dim=1)
        return self.layer_norm(self.adapter(pooled))

class TextAdapter(nn.Module):
    """RoBERTa frozen → adapter → embedding"""
    def __init__(self, d_model=256, bottleneck=256, dropout=0.2):
        super().__init__()
        from transformers import BertModel, BertTokenizer
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for p in self.bert.parameters(): p.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Linear(768, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self._device = nn.Parameter(torch.zeros(1))

    def forward(self, texts):
        d = self._device.device
        if isinstance(texts, str): texts = [texts]
        if not texts or all(t == '' for t in texts):
            return torch.zeros(len(texts), self.adapter[-1].out_features, device=d)
        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        tok = {k: v.to(d) for k, v in tok.items()}
        with torch.no_grad():
            out = self.bert(**tok).last_hidden_state
        m = tok['attention_mask'].unsqueeze(-1).float()
        pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return self.layer_norm(self.adapter(pooled))

class VideoAdapter(nn.Module):
    """OpenFace 49-dim → MLP → adapter → embedding"""
    def __init__(self, d_video=49, d_model=256, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_video, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, video_features, video_mask=None):
        if isinstance(video_features, list):
            max_t = max(v.shape[0] for v in video_features)
            padded, masks = [], []
            for v in video_features:
                pad_len = max_t - v.shape[0]
                padded.append(torch.cat([v, torch.zeros(pad_len, v.shape[1])], dim=0))
                masks.append(torch.cat([torch.ones(v.shape[0]), torch.zeros(pad_len)]))
            video_features = torch.stack(padded)
            video_mask = torch.stack(masks)
        B, T, D = video_features.shape
        x = video_features.reshape(B * T, D)
        x = self.mlp(x).reshape(B, T, -1)
        if video_mask is not None:
            m = video_mask.unsqueeze(-1).to(x.device).float()
            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return self.layer_norm(pooled)

class CrossAttentionFusion(nn.Module):
    """Multi-head cross-attention: each modality attends to others"""
    def __init__(self, d_model=256, n_heads=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, 3, d_model] — audio, text, video embeddings
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x.mean(dim=1)  # [B, d_model]

class DepressionClassifier(nn.Module):
    """Full pipeline: Audio+Text+Video → Cross-Attention → Binary Classifier"""
    def __init__(self, d_model=256, n_classes=2, d_video=49):
        super().__init__()
        self.audio = AudioAdapter(d_model)
        self.text = TextAdapter(d_model)
        self.video = VideoAdapter(d_video=d_video, d_model=d_model)
        self.fusion = CrossAttentionFusion(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, waveform, texts, video_features=None, attention_mask=None, video_mask=None):
        a = self.audio(waveform, attention_mask).unsqueeze(1)   # [B,1,d]
        t = self.text(texts).unsqueeze(1)                        # [B,1,d]
        if video_features is not None:
            v = self.video(video_features, video_mask).unsqueeze(1)  # [B,1,d]
            x = torch.cat([a, t, v], dim=1)  # [B,3,d]
        else:
            x = torch.cat([a, t, torch.zeros_like(a)], dim=1)
        fused = self.fusion(x)
        return self.classifier(fused)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
