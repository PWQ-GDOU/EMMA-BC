#!/usr/bin/env python3
"""
Cross-Cultural Multimodal Adapter Model
Frozen backbones (wav2vec2, BERT, OpenFace) + trainable lightweight adapters.
Stage 1: Pre-train on MODMA (Mandarin) → Stage 2: Fine-tune on DAIC-WOZ (English)
"""
import torch, torch.nn as nn
import numpy as np

class Adapter(nn.Module):
    """Bottleneck adapter: Linear_down → GELU → Dropout → Linear_up"""
    def __init__(self, d_in, d_bottleneck=128, dropout=0.2):
        super().__init__()
        self.down = nn.Linear(d_in, d_bottleneck)
        self.up = nn.Linear(d_bottleneck, d_in)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.dropout(self.up(self.act(self.down(x))))

class AudioAdapter(nn.Module):
    """wav2vec2 (frozen) → adapter → pooled embedding"""
    def __init__(self, d_model=256, bottleneck=128, freeze=True):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.w2v = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        if freeze:
            for p in self.w2v.parameters(): p.requires_grad = False
        self.adapter = Adapter(768, bottleneck)
        self.proj = nn.Linear(768, d_model)

    def forward(self, waveform, attention_mask=None):
        out = self.w2v(waveform, attention_mask=attention_mask).last_hidden_state
        out = self.adapter(out)
        if attention_mask is not None:
            ds = max(1, attention_mask.shape[1] // out.shape[1])
            mask = attention_mask[:, ::ds][:, :out.shape[1]].unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = out.mean(dim=1)
        return self.proj(pooled)

class TextAdapter(nn.Module):
    """BERT (frozen) → adapter → pooled embedding"""
    def __init__(self, d_model=256, bottleneck=128, freeze=True):
        super().__init__()
        from transformers import BertModel, BertTokenizer
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if freeze:
            for p in self.bert.parameters(): p.requires_grad = False
        self.adapter = Adapter(768, bottleneck)
        self.proj = nn.Linear(768, d_model)
        self.device_holder = nn.Parameter(torch.zeros(1))

    def forward(self, texts):
        device = self.device_holder.device
        if isinstance(texts, str): texts = [texts]
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = self.bert(**tokens).last_hidden_state
        out = self.adapter(out)
        mask = tokens['attention_mask'].unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.proj(pooled)

class VideoAdapter(nn.Module):
    """OpenFace features (49-dim) → MLP → adapter → embedding"""
    def __init__(self, d_video=49, d_model=256, bottleneck=64, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_video, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, d_model), nn.GELU(), nn.Dropout(dropout),
        )
        self.adapter = Adapter(d_model, bottleneck, dropout)

    def forward(self, video_features, video_mask=None):
        if isinstance(video_features, list):
            max_t = max(v.shape[0] for v in video_features)
            padded, masks = [], []
            for v in video_features:
                pad_len = max_t - v.shape[0]
                p = torch.cat((v, torch.zeros(pad_len, v.shape[1])), dim=0)
                m = torch.cat([torch.ones(v.shape[0]), torch.zeros(pad_len)])
                padded.append(p)
                masks.append(m)
            video_features = torch.stack(padded)
            video_mask = torch.stack(masks)
        B, T, D = video_features.shape
        x = video_features.reshape(B * T, D)
        x = self.mlp(x).reshape(B, T, -1)
        x = self.adapter(x)
        if video_mask is not None:
            m = video_mask.unsqueeze(-1).to(x.device).float()
            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return pooled

class CrossCulturalModel(nn.Module):
    """Full model: Audio + Text + Video adapters → fusion → classifier"""
    def __init__(self, d_model=256, n_classes=2, d_video=49, freeze_audio=True, freeze_text=True):
        super().__init__()
        self.audio = AudioAdapter(d_model, freeze=freeze_audio)
        self.text = TextAdapter(d_model, freeze=freeze_text)
        self.video = VideoAdapter(d_video=d_video, d_model=d_model)
        self.use_video = True
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, waveform, texts, video_features=None, attention_mask=None, video_mask=None):
        a = self.audio(waveform, attention_mask)
        t = self.text(texts)
        if video_features is not None and self.use_video:
            v = self.video(video_features, video_mask)
            fused = torch.cat([a, t, v], dim=-1)
        else:
            fused = torch.cat([a, t, a * 0], dim=-1)  # keep same dims
        return self.fusion(fused)

    def set_modality(self, mode='full'):
        """full, audio_only, text_only, video_only, audio_video, video_text"""
        self.use_video = mode in ('full', 'video_only', 'audio_video', 'video_text', 'all')
        self._mode = mode

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())
