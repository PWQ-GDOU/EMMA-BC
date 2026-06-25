# EMMA-BC

**E**cological **M**omentary **M**ultimodal **A**ssessment for **B**reast **C**ancer

> 乳腺癌患者多模态生态瞬时评估与症状群数智化管理

## Overview

EMMA-BC is a multimodal deep learning framework for breast cancer patient assessment. It integrates:

- **Audio modality**: wav2vec2-based speech emotion/symptom encoder
- **Text modality**: BERT-based clinical transcript encoder  
- **Fusion**: Cross-modal attention + multi-task regression

## Datasets

| Dataset | Modality | Task | Size |
|---------|----------|------|------|
| RAVDESS | Audio | Emotion classification (6-class) | 1,440 clips |
| DAIC-WOZ | Audio + Text | PHQ-8 regression | 82 participants |
| MODMA | Audio + Clinical | PHQ-9 / GAD-7 / PSQI | 52 participants |

## Project Phases

```
Phase A [DONE]    Audio encoder pretraining (RAVDESS, 67.1% val acc)
Phase B [READY]   Audio + Text -> Clinical regression (DAIC-WOZ + MODMA)
Phase C [PLAN]    Tri-modal fusion + deployment
```

## Architecture

```
Input: Audio (WAV) + Text (transcript)
  |
  +-- AudioEncoder: wav2vec2-base -> TemporalConv -> Transformer -> a
  +-- TextEncoder:  BERT-base -> [CLS] -> Linear -> t
  |
  +-- Fusion: [a; t; a*t; |a-t|] -> MLP
       |
       +-- Head_PHQ:  Depression severity
       +-- Head_GAD:  Anxiety severity  
       +-- Head_PSQI: Sleep quality
```

## Setup

```bash
git clone https://github.com/PWQ-GDOU/EMMA-BC.git
cd EMMA-BC
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Citation

```bibtex
@software{emma-bc,
  title = {EMMA-BC: Ecological Momentary Multimodal Assessment for Breast Cancer},
  author = {PWQ-GDOU},
  year = {2026},
  url = {https://github.com/PWQ-GDOU/EMMA-BC}
}
```
