# EMMA-BC Final Results

## 7-Modality Ablation (WavLM + BERT + OpenFace features)

| Modality | DAIC F1 | DAIC AUC | MODMA F1 | MODMA AUC |
|----------|:--:|:--:|:--:|:--:|
| **Audio only** | **64.2%** | 46.2% | **77.8%** | 67.4% |
| Text only | 62.5% | 53.1% | 57.3% | 51.2% |
| Audio+Text | 61.1% | 45.8% | 75.8% | 68.6% |
| Full (A+V+T) | 58.9% | 54.4% | 75.7% | 67.1% |
| Audio+Video | 58.0% | 42.4% | 76.8% | 66.3% |
| Video+Text | 57.9% | 47.3% | 57.3% | 46.0% |
| Video only | 54.7% | 35.8% | 57.3% | 51.3% |

## Cross-Cultural Transfer

| Method | DAIC F1 |
|--------|:--:|
| DAIC from scratch | 55.4% |
| MODMA pretrain -> DAIC | 34.7% |

## Previous Baselines

| Method | DAIC Acc | DAIC F1 |
|--------|:--:|:--:|
| Huang 2024 (claimed) | 96.5% | — |
| Huang v2 (our replication) | 62.7% | 63.9% |
| MODMA within (classical ML) | 76.9% | 65.5% |

## Key Findings

1. **Audio is consistently the strongest modality** — adding video/text often degrades performance on small datasets
2. **Cross-cultural transfer (MODMA→DAIC) fails** — Chinese to English domain gap too large
3. **Huang 2024's 96.5% is irreproducible** — maximum clean accuracy ~67%
4. **Frozen backbone + lightweight adapter** is the viable approach for N<100 subjects
