# Data Preparation Guide

## 1. RAVDESS (Phase A: Audio Emotion Pretraining)
- **Source**: https://zenodo.org/record/1188976
- **Path**: `/data/disk1/datasets/ravdess/`
- **Structure**: 24 actor folders, 1,440 WAV files total
- **Labels**: Encoded in filename (position 3 = emotion code)
- **Used emotions**: neutral(01), happy(03), sad(04), angry(05), fearful(06), disgust(07)

## 2. DAIC-WOZ (Phase B: Clinical Regression)
- **Source**: https://dcapswoz.ict.usc.edu/wwwedaic/ (license required)
- **Path**: `/data/disk1/datasets/diac_woz/`
- **Structure**:
  ```
  diac_woz/
  ├── {300-382}_P.tar.gz    # 82 participant archives (~30GB)
  ├── labels2019.tar.gz      # PHQ-8 labels + train/dev/test split
  └── extracted/{pid}/
      ├── {pid}_AUDIO.wav    # Clinical interview audio (~20MB)
      └── {pid}_Transcript.csv
  ```
- **Labels**: PHQ-8 (8 items + total, 0-24)
- **Splits**: Official train/dev/test in `labels/{train,dev,test}_split.csv`

## 3. MODMA (Phase B: Clinical Regression)
- **Source**: https://modma.lzu.edu.cn/ (license required)
- **Path**: `/data/disk1/datasets/modma/`
- **Structure**:
  ```
  modma/
  ├── audio_lanzhou_2015.zip          # Speech (2.5GB)
  └── extracted/audio_lanzhou_2015/
      ├── {subject_id}/               # 52 subjects
      │   └── {01-28}.wav             # ~27 recordings/subject
      └── subjects_information_audio_lanzhou_2015.xlsx
  ```
- **Labels**: PHQ-9 (0-27), GAD-7 (0-21), PSQI (0-21), CTQ-SF, LES, SSRS

## Environment Variables
```bash
export EMMA_RAVDESS_DIR=/path/to/ravdess
export EMMA_DAIC_DIR=/path/to/diac_woz
export EMMA_MODMA_DIR=/path/to/modma
```
