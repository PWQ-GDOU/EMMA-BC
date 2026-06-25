# EMMA-BC: Ecological Momentary Multimodal Assessment for Breast Cancer

> 乳腺癌患者多模态生态瞬时评估与症状群数智化管理
>
> **PI**: 班悦 (Yue Ban) — 广东医科大学
>
> [![Phase A](https://img.shields.io/badge/Phase_A-training-green)](#) [![Phase B](https://img.shields.io/badge/Phase_B-ready-blue)](#) [![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org) [![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-red)](https://pytorch.org)

---

## Overview

EMMA-BC is a multimodal deep learning framework that integrates **speech audio** and **clinical transcripts** for depression severity assessment (PHQ-8 regression) in breast cancer patients. The system follows a three-phase research pipeline: pretraining → clinical regression → deployment.

### Clinical Motivation

Traditional depression screening relies on self-report questionnaires (PHQ-8/PHQ-9). EMMA-BC augments this with passive, speech-based biomarkers extracted from clinical interviews — enabling objective, continuous monitoring without additional patient burden.

### Modalities

| Modality | Encoder | Input | Output |
|----------|---------|-------|--------|
| Audio | wav2vec2-base + 4-layer Transformer | Raw 16kHz WAV | 256d emotion-aware embedding |
| Text | BERT-base (frozen) + Linear | Clinical transcript | 256d semantic embedding |
| Fusion | [audio; text; audio⊙text; |audio−text|] → MLP | Concatenated cross-modal features | PHQ-8 score |

---

## Project Structure

```
EMMA-BC/
├── phaseA_pretrain.py              # Phase A: Audio emotion encoder (RAVDESS)
├── phaseA_augment.py               # [DEPRECATED] Data augmentation route
├── phaseB/
│   ├── multimodal_model.py         # wav2vec2 + BERT fusion architecture
│   ├── multimodal_dataset.py       # DAICWOZDataset + MODMADataset
│   └── phaseB_train.py             # Multi-task clinical regression training
├── analysis/
│   ├── interpret.py                # Permutation importance + error analysis + fairness
│   ├── clinical_eval.py            # Binary classification + MC-Dropout CI + fallback
│   └── table_generator.py          # Paper-ready table material generation
├── configs/
│   ├── mult_pretrain.yaml          # Phase A config
│   └── scales/                     # Clinical scale schemas (HADS, VAS, CFS, etc.)
├── docs/                           # Project documentation (Chinese)
├── requirements.txt
├── DATA_COMPLIANCE.md              # Dataset license & compliance
├── DATA.md
├── STRUCTURE.md
└── README.md
```

---

## Quick Start

### Prerequisites

```bash
git clone https://github.com/PWQ-GDOU/EMMA-BC.git
cd EMMA-BC
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com  # China users
```

### Datasets

| Dataset | Modality | Task | Participants | Source |
|---------|----------|------|-------------|--------|
| **RAVDESS** | Audio | 6-class emotion classification | 24 actors | CC BY-NC-SA 4.0 |
| **DAIC-WOZ** | Audio + Text | PHQ-8 regression | 82 | USC (restricted) |
| **MODMA** | Audio + EEG + Clinical | PHQ-9 / GAD-7 / PSQI | 52 | MODMA (license required) |

Place datasets under `/data/disk1/datasets/` (or override with `--data`).

### Phase A: Audio Encoder Pretraining

```bash
CUDA_VISIBLE_DEVICES=0 python phaseA_pretrain.py \
    --epochs 30 --batch_size 16 --lr 1e-4
```

**Expected result**: ~67% validation accuracy on 6-class RAVDESS emotion classification.  
**Output**: `checkpoints/phaseA/phaseA_best.pt`

### Phase B: Multimodal Clinical Regression

```bash
CUDA_VISIBLE_DEVICES=0 python phaseB/phaseB_train.py \
    --audio_pretrained checkpoints/phaseA/phaseA_best.pt \
    --data /data/disk1/datasets/diac_woz \
    --batch_size 2 --gradient_accumulation_steps 4 \
    --lr 5e-4 --epochs 100 --patience 10
```

**Key parameters for 11GB GPU (RTX 2080 Ti)**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--batch_size` | 2 | wav2vec2 + BERT in 11GB |
| `--gradient_accumulation_steps` | 4 | Effective batch size = 8 |
| `--lr` | 5e-4 | Optimized for small fusion head |
| `--patience` | 10 | Early stopping |
| `--filter_interviewer` | False | Use full transcript for evaluation |

### Session Management

```bash
# Launch in background (survives SSH disconnect)
tmux new-session -d -s phaseB 'cd EMMA-BC && source venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python phaseB/phaseB_train.py ... 2>&1 | tee logs/phaseB.log'

# Reattach
tmux attach -t phaseB
```

---

## Analysis Tools

After training, generate paper-ready results:

```bash
# 1. Model interpretability
python analysis/interpret.py --checkpoint checkpoints/phaseB/phaseB_best.pt

# 2. Clinical evaluation (classification + uncertainty)
python analysis/clinical_eval.py --checkpoint checkpoints/phaseB/phaseB_best.pt --mc_samples 30

# 3. Predict CSV with confidence scores
# (call predict_to_csv from clinical_eval)

# 4. Paper table material
python analysis/table_generator.py --csv checkpoints/predictions.csv
```

| Tool | Output |
|------|--------|
| `interpret.py` | Permutation importance (audio vs. text), error case analysis, subgroup fairness |
| `clinical_eval.py` | Binary classification (sensitivity/specificity/F1), MC-Dropout 95% CI, missing-modality fallback |
| `table_generator.py` | 4 tables: Regression, Classification, Trust Score, Modality Breakdown |

### Prediction CSV Columns

```
Subject_ID, PHQ_Score, Confidence, Modality_Status,
MC_Mean, MC_Std, CI_Lower_95, CI_Upper_95,
Fused_Trust_Score, Binary_Label
```

---

## Engineering Safeguards

The codebase implements 30+ safeguards against common research code pitfalls:

| Category | Safeguards |
|----------|------------|
| **Data Leakage** | Split by `participant_id` (not random shuffle); official DAIC-WOZ train/test split protocol |
| **Reproducibility** | `set_seed(42)` + `cudnn.deterministic=True`; optimizer/scheduler/normalizer state in checkpoints |
| **Numerical Stability** | `clamp(min=1e-6)` in pooling; grouped weight decay (no decay on bias/LN); Z-score normalization + inverse evaluation |
| **Clinical Validity** | PHQ-8 ≥ 10 binary classification; percentile-based 95% CI (not Gaussian); exponential-decay trust score |
| **Training Robustness** | GPU-only (`cuda:0`); gradient clipping `1.0`; `mp.set_start_method('spawn')`; early stopping |
| **Shortcut Prevention** | Interviewer prompt filter; short audio filter (`min_audio_sec=1.0`); NaN label tolerance for test set |
| **Resume Support** | Epoch/optimizer/scheduler/normalizer/`best_mae`/`best_ccc` — full state restoration |

---

## Known Limitations

1. **DAIC-WOZ Transcripts**: No speaker labels — interviewer (Ellie) prompts interleave with patient speech. `filter_interviewer` option available but may false-positive on patient quotes.
2. **Single-Modality Fallback**: Audio-only and text-only predictions are not trained — fallback is demonstration-only. **Use complete dual-modality samples for paper evaluation.**
3. **MODMA**: Currently unsupported in Phase B training pipeline; dataset available for future multi-site validation.
4. **Test Set Labels**: DAIC-WOZ official test labels are hidden. Submit predictions to EvalAI for official scoring — never report local test metrics.

---

## Citation

```bibtex
@software{emma-bc-2026,
  title        = {EMMA-BC: Ecological Momentary Multimodal Assessment for Breast Cancer},
  author       = {PWQ-GDOU},
  year         = {2026},
  url          = {https://github.com/PWQ-GDOU/EMMA-BC},
  note         = {PI: Yue Ban, Guangdong Medical University}
}
```
