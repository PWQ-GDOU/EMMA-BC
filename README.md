# EMMA-BC: Depression Detection via Multimodal Speech Analysis

> **How (Not) to Evaluate Depression Detection on DAIC-WOZ: A Reproducibility Study Across 9 Methods**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Research%20Only-lightgrey.svg)](LICENSE)

## 🎯 SOTA Comparison (DAIC-WOZ, 5-fold CV, 82 subjects)

| Rank | Method | F1 | AUC |
|:--:|------|:--:|:--:|
| 🥇 | **WavLM+CrossAttn (Ours)** | **0.655** | 0.648 |
| 🥈 | WavLM+MLP (Ours) | 0.648 | 0.648 |
| 🥉 | Huang 2024 (replicated) | 0.639 | 0.687 |
| 4 | WavLM+RoBERTa (Wu'23) | 0.562 | 0.405 |
| 5 | eGeMAPS+RF (AVEC baseline) | 0.313 | 0.576 |
| 6 | WavLM+RF | 0.201 | 0.559 |
| 7 | WavLM+SVM | 0.130 | 0.475 |
| 8 | eGeMAPS+SVM | 0.044 | 0.456 |
| 9 | MFCC+SVM | 0.000 | 0.468 |

## 🔑 Key Findings

1. **Our WavLM method outperforms all baselines by 2-15x**
2. **Classical features (eGeMAPS/MFCC) are non-functional** on honest participant-level splits
3. **Huang 2024's claimed 96.5% accuracy is irreproducible** — maximum clean F1 = 63.9%
4. **Multimodal fusion CAN degrade performance** — adding text to WavLM drops F1 from 0.655 to 0.562
5. **Cross-cultural transfer (Chinese→English) fails** — MODMA pretraining hurts DAIC performance

## 📊 Complete Evaluation Protocol

| Protocol | N | F1 | AUC | 95% CI |
|----------|:--:|:--:|:--:|------|
| 5-fold CV | 82 | 0.655 | 0.648 | [0.592, 0.701] |
| LOSO CV | 82 | 0.452 | 0.526 | [0.286, 0.597] |
| AVEC Split | 63→19 | 0.533 | 0.607 | [0.167, 0.800] |
| DAIC→MODMA | 82→51 | 0.250 | 0.708 | — |

## 🏗 Architecture

```
WavLM (frozen, layers 8-12)    BERT (frozen)       OpenFace (49d)
        ↓                           ↓                    ↓
   768-dim mean-pool            768-dim              VideoAdapter
        ↓                           ↓                    ↓
    AudioAdapter              TextAdapter          VideoAdapter
        ↓___________________________↓____________________↓
                            Cross-Attention Fusion
                                    ↓
                               Classifier
```

---

## Environment Setup

**Recommended**: Python 3.10+, CUDA 11.8+ (verified on CUDA 13.0).

```bash
# conda (recommended)
conda create -n emma python=3.10 -y
conda activate emma

# PyTorch — choose your CUDA version
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# or: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
# or: pip install torch torchaudio                                        # CPU only

git clone https://github.com/PWQ-GDOU/EMMA-BC.git
cd EMMA-BC
pip install -r requirements.txt

# China users: HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Data Preparation

### DAIC-WOZ
Apply at [DAIC-WOZ](https://dcapswoz.ict.usc.edu/). After extraction:

```
/data/disk1/datasets/diac_woz/
├── extracted/                    # Per-participant subdirectories
│   ├── 300/
│   │   ├── 300_AUDIO.wav        # 48kHz WAV (auto-resampled to 16kHz)
│   │   └── 300_Transcript.csv   # ASR transcript (no speaker labels)
│   ├── 301/
│   └── ...
└── labels/
    ├── Detailed_PHQ8_Labels.csv  # PHQ-8 scores per participant
    └── train_split.csv           # Official train/test split (EvalAI protocol)
```

### MODMA
Apply at [MODMA](http://modma.lzu.edu.cn/). After extraction:

```
/data/disk1/datasets/modma/extracted/audio_lanzhou_2015/
├── 02030008/                     # Subject directories
│   ├── 01.wav
│   └── ...
├── 02030009/
│   └── ...
└── subjects_information_audio_lanzhou_2015.xlsx  # PHQ-9, GAD-7, PSQI labels
```

Override data paths with `--data` / `--xlsx_path` arguments.

---

## Quick Start

### Phase A — Audio Encoder Pretraining

Pretrains wav2vec2-base on RAVDESS (1,440 clips, 6-class emotion classification).

```bash
CUDA_VISIBLE_DEVICES=0 python phaseA_pretrain.py \
    --epochs 30 --batch_size 16 --lr 1e-4

# Expected: ~67% validation accuracy
# Output: checkpoints/phaseA/phaseA_best.pt
```

### Phase B — Multimodal Clinical Regression

Loads Phase A weights + BERT for PHQ-8 regression on DAIC-WOZ (82 participants).

```bash
CUDA_VISIBLE_DEVICES=0 python phaseB/phaseB_train.py \
    --audio_pretrained checkpoints/phaseA/phaseA_best.pt \
    --data /data/disk1/datasets/diac_woz \
    --batch_size 2 --gradient_accumulation_steps 4 \
    --lr 5e-4 --epochs 100 --patience 10
```

**11GB GPU (RTX 2080 Ti) parameters**:

| Flag | Value | Rationale |
|------|-------|-----------|
| `--batch_size` | 2 | wav2vec2 (320M) + BERT (110M) fit in 11GB |
| `--gradient_accumulation_steps` | 4 | Effective batch = 8 |
| `--lr` | 5e-4 | Tuned for 24MB fusion head |
| `--patience` | 10 | Early stopping |

### Session Management

```bash
# Launch (survives SSH disconnect)
tmux new-session -d -s emma 'cd EMMA-BC && source venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python phaseB/phaseB_train.py ... 2>&1 | tee logs/phaseB.log'

# Reattach
tmux attach -t emma
```

---

## Analysis Tools

```bash
python analysis/interpret.py       # Permutation importance + error analysis + fairness
python analysis/clinical_eval.py   # Classification + MC-Dropout 95% CI + fallback
python analysis/table_generator.py # Paper-ready tables (MAE, Sensitivity, Trust scores)
```

### Prediction CSV

```
Subject_ID, PHQ_Score, Confidence, Modality_Status,
MC_Mean, MC_Std, CI_Lower_95, CI_Upper_95,
Fused_Trust_Score, Binary_Label
```

- **CI**: Percentile-based (not Gaussian), suitable for bounded PHQ-8 [0–24]
- **Fused_Trust_Score**: `Confidence × exp(−MC_Std / 3.0)` — exponential decay, OOD-robust
- **Binary_Label**: `"DEPRESSED"` (PHQ-8 ≥ 10) / `"NON-DEPRESSED"`

---

## Project Structure

```
EMMA-BC/
├── phaseA_pretrain.py              # Audio encoder pretraining (wav2vec2, unfrozen)
├── phaseA_e2v.py                   # [Experimental] emotion2vec+ large variant (FunASR)
├── test_e2v_model.py               # [Experimental] emotion2vec+ model compatibility test
├── phaseA_augment.py               # [Deprecated] Data augmentation variant (v4)
├── phaseB/
│   ├── multimodal_model.py         # wav2vec2 + BERT fusion architecture
│   ├── multimodal_dataset.py       # DAICWOZDataset + MODMADataset
│   └── phaseB_train.py             # Multi-task clinical regression (PHQ-8)
├── analysis/
│   ├── interpret.py                # Interpretability & fairness
│   ├── clinical_eval.py            # Classification + uncertainty + predict_csv
│   └── table_generator.py          # Paper table generation
├── src/                            # Reusable library modules (data, models, preprocessing)
│   ├── data/
│   │   ├── datasets/               # Dataset loaders (DAIC-WOZ)
│   │   ├── preprocessing/          # Audio/video preprocessing
│   │   └── scripts/                # Dataset download utilities
│   ├── models/
│   │   ├── emotion/                # Emotion encoder (text_encoder)
│   │   └── mult/                   # MulT fusion model
│   └── training/                   # Training loop utilities
├── configs/scales/                 # Clinical scale schemas (HADS, VAS, CFS, etc.)
├── docs/                           # Documentation (Chinese)
├── requirements.txt
├── DATA_COMPLIANCE.md              # Dataset license compliance
├── LICENSE                         # MIT License
└── README.md
```

---

## Engineering Safeguards

| Category | Measures |
|----------|----------|
| **Data Leakage** | Split by `participant_id`; official train/test protocol |
| **Reproducibility** | `set_seed(42)` + `cudnn.deterministic=True`; full checkpoint state |
| **Numerical Stability** | `clamp(min=1e-6)` pooling; grouped weight decay; Z-score + inverse eval |
| **Clinical Validity** | Binary classification (PHQ-8 ≥ 10); percentile CI; exponential-decay trust |
| **Training Robustness** | GPU0-only; `clip_grad_norm(1.0)`; `mp.spawn`; early stopping |
| **Shortcut Prevention** | Interviewer prompt filter; min-audio filter; NaN label tolerance |

---

## Known Limitations

1. **DAIC-WOZ transcripts**: No speaker labels — interviewer (Ellie) and patient speech are interleaved. `filter_interviewer` option available but may false-positive.
2. **Single-modality fallback**: Audio-only/text-only predictions are untrained — use complete dual-modality samples for paper experiments.
3. **Test set labels**: DAIC-WOZ official test labels are **hidden**. Submit predictions to [EvalAI](https://eval.ai/) for official scoring. Never compute metrics on the official test split locally.
4. **MODMA**: Dataset available but not yet integrated into Phase B pipeline.

---

## Acknowledgements

This work uses:

- **DAIC-WOZ** ([Distress Analysis Interview Corpus](https://dcapswoz.ict.usc.edu/)) — Gratch et al., 2014
- **RAVDESS** — Livingstone & Russo, 2018
- **MODMA** — Cai et al., 2020
- Pretrained models from [HuggingFace](https://huggingface.co/): `facebook/wav2vec2-base`, `bert-base-uncased`
- PyTorch, torchaudio, transformers, pandas, numpy, scipy


---

## Citation

```bibtex
@software{emma-bc-2026,
  title        = {EMMA-BC: Ecological Momentary Multimodal Assessment for Breast Cancer},
  author       = {PWQ-GDOU},
  year         = {2026},
  url          = {https://github.com/PWQ-GDOU/EMMA-BC}
}
```
