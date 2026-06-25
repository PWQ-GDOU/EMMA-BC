# Code Structure

## Current
```
EMMA-BC/
├── phaseA_pretrain.py         # Phase A: audio emotion encoder (RAVDESS)
├── phaseA_augment.py          # [DEPRECATED] augmentation experiment
├── phaseB/
│   ├── multimodal_model.py    # wav2vec2 + BERT fusion model
│   └── multimodal_dataset.py  # DAIC-WOZ + MODMA data loaders
├── src/                       # Legacy/early prototypes
│   ├── models/
│   │   ├── emotion/           # Early emotion model prototypes
│   │   └── mult/              # MulT model reference
│   └── training/              # Legacy training utilities
├── configs/                   # YAML configs + scale JSON definitions
├── docs/                      # Project documentation
└── requirements.txt           # Python dependencies
```

## Planned (TODO)
```
EMMA-BC/
├── emma/                      # Main package
│   ├── models/
│   │   ├── audio.py           # AudioEncoder
│   │   ├── text.py            # TextEncoder
│   │   └── fusion.py          # MultimodalFusion
│   ├── data/
│   │   ├── daic.py            # DAICWOZDataset
│   │   ├── modma.py           # MODMADataset
│   │   └── ravdess.py         # EmotionDataset
│   └── utils/
│       ├── seed.py            # set_seed()
│       └── audio.py           # Audio preprocessing helpers
├── scripts/
│   ├── train_phaseA.py
│   └── train_phaseB.py
└── configs/
```
Both `phaseB/` and `src/` paths work — choose one for consistency.
