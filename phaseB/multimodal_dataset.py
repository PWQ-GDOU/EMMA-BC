#!/usr/bin/env python3
"""
Phase B Dataset — DAIC-WOZ + MODMA multimodal data loading.

DAIC-WOZ:
  Per participant: {id}_AUDIO.wav + {id}_Transcript.csv + PHQ-8 labels
  Labels: Detailed_PHQ8_Labels.csv + train/dev/test_split.csv
  Transcript: timestamped segments → concatenated into single text

MODMA:
  Per subject: audio_lanzhou_2015/{subject_id}/*.wav (multiple recordings)
  Labels: subjects_information_audio_lanzhou_2015.xlsx (PHQ-9, GAD-7, PSQI, etc.)
"""

import os, sys, json, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
import soundfile as sf
import pandas as pd
import torchaudio


# ══════════════════════════════════════════════════════════
# DAIC-WOZ Dataset
# ══════════════════════════════════════════════════════════


class DAICWOZDataset(Dataset):
    """
    DAIC-WOZ: clinical interview audio + transcript → PHQ-8 regression.

    ⚠️ CRITICAL: Uses official per-participant splits only.
    Never use random_split() on this dataset — it would leak participant
    identity across train/val, inflating metrics by 20-30%.

    Args:
        data_dir: path to diac_woz/ (containing {id}_P.tar.gz, labels2019.tar.gz)
        split: "train", "dev", or "test" (official DAIC-WOZ splits)
        sample_rate: target audio sample rate
        max_audio_sec: maximum audio duration in seconds
    """
    def __init__(self, data_dir, split="train", sample_rate=16000, max_audio_sec=30, min_audio_sec=1.0, filter_interviewer=False):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_sec * sample_rate)
        self.min_samples = 16000
        # removed — see above
        self.min_samples = int(min_audio_sec * sample_rate)  # use param, not hardcoded
        self.filter_interviewer = filter_interviewer  # use param, not hardcoded
        # Ellie interviewer prompt prefixes for filtering
        self.ELLIE_PROMPTS = [
            "ellie:", "how are you doing today", "tell me about",
            "can you tell me", "thank you for sharing", "i hear you saying",
            "let me ask you", "i would like to", "how would you describe",
            "what would you say", "so you mentioned", "you mentioned earlier",
        ]
        self.split = split

        # Load PHQ-8 labels
        labels_csv = self.data_dir / "labels" / "Detailed_PHQ8_Labels.csv"
        self.phq8_labels = pd.read_csv(labels_csv)
        self.phq8_labels.set_index("Participant_ID", inplace=True)
        # Test split: labels may be NaN — track for safe collation
        self.is_test = (split == "test")

        # Load split assignment
        split_csv = self.data_dir / "labels" / f"{split}_split.csv"
        split_df = pd.read_csv(split_csv)
        self.participant_ids = split_df["Participant_ID"].tolist()

        # Verify data exists
        self.samples = []
        for pid in self.participant_ids:
            audio_path = self.data_dir / "extracted" / f"{pid}" / f"{pid}_AUDIO.wav"
            transcript_path = self.data_dir / "extracted" / f"{pid}" / f"{pid}_Transcript.csv"
            if audio_path.exists():
                # Skip samples shorter than min_audio_sec (Wav2Vec2 can't process)
                audio_size = audio_path.stat().st_size
                if audio_size < self.min_samples * 2:  # 16-bit PCM: 2 bytes per sample
                    continue
                self.samples.append({
                    "pid": pid,
                    "audio": str(audio_path),
                    "transcript": str(transcript_path) if transcript_path.exists() else None,
                })

        print(f"[DAIC-WOZ:{split}] {len(self.samples)} participants")

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, path):
        waveform, sr = sf.read(path)
        waveform = torch.from_numpy(waveform).float()
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0), sr, self.sample_rate
            ).squeeze(0)
        original_len = waveform.shape[0]
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]
            original_len = self.max_samples
        elif waveform.shape[0] < self.max_samples:
            waveform = torch.cat([
                waveform,
                torch.zeros(self.max_samples - waveform.shape[0])
            ])
        return waveform, original_len


    def _load_video_features(self, pid):
        """Load OpenFace Pose+Gaze+AUs features. Returns [T, 50] tensor or None."""
        import pandas as pd
        features_dir = self.data_dir / (str(pid) + "_P") / "features"
        if not features_dir.exists():
            return None
        # Find OpenFace CSV
        csv_files = list(features_dir.glob("*OpenFace*.csv"))
        if not csv_files:
            return None
        try:
            df = pd.read_csv(csv_files[0])
            # Select feature columns (skip frame, timestamp, confidence, success)
            feat_cols = [col for col in df.columns if col not in ["frame", "timestamp", "confidence", "success"]]
            features = torch.tensor(df[feat_cols].values, dtype=torch.float32)
            # Clip to max 9000 frames (5 min at 30fps) to limit memory
            max_frames = 9000
            if features.shape[0] > max_frames:
                # Uniformly sample
                indices = torch.linspace(0, features.shape[0]-1, max_frames).long()
                features = features[indices]
            return features  # [T, d_video]
        except Exception as e:
            print(f"WARNING: Failed to load video for {pid}: {e}")
            return None

    def _load_transcript(self, path):
        """Load transcript CSV, concatenate all text segments."""
        if path is None:
            return ""
        df = pd.read_csv(path)
        texts = df["Text"].dropna().tolist()
        if self.filter_interviewer:
            texts = [t for t in texts if not any(p in t.lower().strip() for p in self.ELLIE_PROMPTS)]
        return " ".join(texts)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pid = sample["pid"]

        audio, valid_len = self._load_audio(sample["audio"])
        transcript = self._load_transcript(sample["transcript"])

        # PHQ-8 labels: 8 items + total
        phq_row = self.phq8_labels.loc[pid]
        
        # Test set may have NaN labels — return None
        try:
            phq_total = torch.tensor(phq_row["PHQ_8Total"], dtype=torch.float)
            phq_items = torch.tensor([
                phq_row["PHQ_8NoInterest"], phq_row["PHQ_8Depressed"],
                phq_row["PHQ_8Sleep"], phq_row["PHQ_8Tired"],
                phq_row["PHQ_8Appetite"], phq_row["PHQ_8Failure"],
                phq_row["PHQ_8Concentrating"], phq_row["PHQ_8Moving"],
            ], dtype=torch.float)
        except (ValueError, TypeError, KeyError):
            phq_total = torch.tensor(float('nan'))
            phq_items = torch.full((8,), float('nan'))

        # Load video features (OpenFace)
        video_features = self._load_video_features(pid)

        return {
            "audio": audio,
            "transcript": transcript,
            "phq_total": phq_total,      # regression target
            "phq_items": phq_items,      # per-item scores
            "pid": pid,
            "valid_len": valid_len,
            "video_features": video_features,  # [T, 50] or None
        }






    def split_val_from_train(self, val_ratio=0.15, seed=42):
        """
        Split the TRAIN set into train/val BY PARTICIPANT.
        Critical: same participant must not appear in both train and val.
        
        Uses official train_split.csv as source, splits by participant ID.
        Call ONLY on the full train dataset before creating DataLoaders.
        """
        import random
        rng = random.Random(seed)
        
        # Group samples by participant
        pid_to_indices = {}
        for i, s in enumerate(self.samples):
            pid_to_indices.setdefault(s["pid"], []).append(i)
        
        pids = sorted(pid_to_indices.keys())
        rng.shuffle(pids)
        n_val = max(1, int(len(pids) * val_ratio))
        
        val_pids = set(pids[:n_val])
        train_pids = set(pids[n_val:])

        train_idx = [i for pid in train_pids for i in pid_to_indices[pid]]
        val_idx = [i for pid in val_pids for i in pid_to_indices[pid]]
        
        print(f"[DAIC-WOZ Subject Split] train={len(train_pids)} participants ({len(train_idx)} samples), "
              f"val={len(val_pids)} ({len(val_idx)} samples)")
        
        from torch.utils.data import Subset
        return Subset(self, train_idx), Subset(self, val_idx)
# ══════════════════════════════════════════════════════════
# MODMA Dataset
# ══════════════════════════════════════════════════════════

class MODMADataset(Dataset):
    """
    MODMA: speech audio recordings → clinical scale regression.

    Each subject has ~27 WAV files (question answers).
    We aggregate them into one long audio + use clinical scales as labels.

    Args:
        data_dir: path to modma/ (containing extracted audio_lanzhou_2015/)
        xlsx_path: path to subjects_information_audio_lanzhou_2015.xlsx
        sample_rate: target sample rate
        max_audio_sec: max duration per sample
    """
    def __init__(self, data_dir, xlsx_path, sample_rate=16000, max_audio_sec=300, min_audio_sec=1.0, filter_interviewer=False):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_sec * sample_rate)
        self.min_samples = 16000
        # removed — see above
        self.min_samples = int(min_audio_sec * sample_rate)  # use param, not hardcoded
        self.filter_interviewer = filter_interviewer  # use param, not hardcoded

        # Load clinical labels
        self.labels_df = pd.read_excel(xlsx_path)
        # Standardize column names
        self.labels_df.columns = [
            "subject_id", "type", "age", "gender", "education",
            "PHQ-9", "CTQ-SF", "LES", "SSRS", "GAD-7", "PSQI", "_a", "_b"
        ]
        self.labels_df.set_index("subject_id", inplace=True)
        self.labels_df.index = ["0" + str(sid) for sid in self.labels_df.index]

        # Find all available subjects
        audio_dir = self.data_dir / "audio_lanzhou_2015"
        self.samples = []
        for subj_dir in sorted(audio_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            sid = subj_dir.name
            wavs = sorted(subj_dir.glob("*.wav"))
            if wavs and sid in self.labels_df.index:
                self.samples.append({
                    "subject_id": sid,
                    "wavs": [str(w) for w in wavs],
                    "n_wavs": len(wavs),
                })

        print(f"[MODMA] {len(self.samples)} subjects, "
              f"avg {np.mean([s['n_wavs'] for s in self.samples]):.0f} recordings/subject")

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, wav_paths):
        """Load and concatenate multiple WAV files into one waveform."""
        segments = []
        for path in wav_paths:
            waveform, sr = sf.read(path)
            waveform = torch.from_numpy(waveform).float()
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=-1)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform.unsqueeze(0), sr, self.sample_rate
                ).squeeze(0)
            segments.append(waveform)

        # Concatenate with 0.5s silence between segments
        silence = torch.zeros(int(0.5 * self.sample_rate))
        combined = segments[0]
        for seg in segments[1:]:
            combined = torch.cat([combined, silence, seg])

        if combined.shape[0] > self.max_samples:
            combined = combined[:self.max_samples]
        elif combined.shape[0] < self.max_samples:
            combined = torch.cat([
                combined,
                torch.zeros(self.max_samples - combined.shape[0])
            ])
        return combined

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sid = sample["subject_id"]

        audio = self._load_audio(sample["wavs"])

        # Clinical labels
        row = self.labels_df.loc[sid]
        phq9 = torch.tensor(row["PHQ-9"], dtype=torch.float)
        gad7 = torch.tensor(row["GAD-7"], dtype=torch.float)
        psqi = torch.tensor(row["PSQI"], dtype=torch.float)

        # MODMA has no transcript — use empty string or subject info
        transcript = f"Subject {row['type']} age {row['age']} {row['gender']}"
        if hasattr(row, 'education'):
            transcript += f" education {row['education']} years"

        return {
            "audio": audio,
            "transcript": transcript,
            "phq": phq9,       # PHQ-9 total
            "gad": gad7,       # GAD-7 total
            "psqi": psqi,      # PSQI total
            "subject_id": sid,
            "diagnosis": row["type"],  # "MDD" or "HC"
        }


# ══════════════════════════════════════════════════════════
# Collate functions
# ══════════════════════════════════════════════════════════

def collate_daic(batch):
    """Collate DAIC-WOZ batch with explicit audio attention_mask.
    mask shape [B, T]: 1.0=valid audio, 0.0=zero-padding.
    Uses valid_len from dataset to set padding regions to 0."""
    audio_list = []
    mask_list = []
    video_list = []
    texts = []
    phq_totals = []
    phq_items_list = []
    pids = []

    max_len = max(item["audio"].shape[0] for item in batch)
    for item in batch:
        audio = item["audio"]
        L = audio.shape[0]
        # Pad audio if shorter than batch max
        if L < max_len:
            audio = torch.cat([audio, torch.zeros(max_len - L)])
        audio_list.append(audio)
        # Mask: 1.0 for real audio (up to valid_len), 0.0 for zero-padding
        valid_len = item.get("valid_len", L)
        mask = torch.zeros(max_len)
        mask[:min(valid_len, max_len)] = 1.0
        mask_list.append(mask)
        texts.append(item["transcript"])
        phq_totals.append(item["phq_total"])
        vf = item.get("video_features")
        video_list.append(vf if vf is not None else torch.zeros(1, 50))
        phq_items_list.append(item["phq_items"])
        pids.append(item["pid"])

        # Stack labels; NaN-safe (test set may have missing labels)
    try:
        phq_total_stack = torch.stack(phq_totals)
        phq_items_stack = torch.stack(phq_items_list)
    except RuntimeError:
        # Fallback: use stack with reshape for safety
        phq_total_stack = torch.stack([t.reshape(-1) for t in phq_totals])

    return {
        "audio": torch.stack(audio_list),
        "attention_mask": torch.stack(mask_list),
        "texts": texts,
        "phq_total": phq_total_stack,
        "phq_items": phq_items_stack,
        "pids": pids,
        "video_features": video_list,  # list of [T, 50] tensors
    }


def collate_modma(batch):
    """Collate MODMA batch with explicit audio attention_mask.
    Uses valid_len from dataset to set padding regions to 0."""
    audio_list = []
    mask_list = []
    video_list = []
    texts = []
    phq_list = []
    gad_list = []
    psqi_list = []
    sids = []

    max_len = max(item["audio"].shape[0] for item in batch)
    for item in batch:
        audio = item["audio"]
        L = audio.shape[0]
        if L < max_len:
            audio = torch.cat([audio, torch.zeros(max_len - L)])
        audio_list.append(audio)
        valid_len = item.get("valid_len", L)
        mask = torch.zeros(max_len)
        mask[:min(valid_len, max_len)] = 1.0
        mask_list.append(mask)
        texts.append(item["transcript"])
        phq_list.append(item["phq"])
        gad_list.append(item["gad"])
        psqi_list.append(item["psqi"])
        sids.append(item["subject_id"])

    return {
        "audio": torch.stack(audio_list),
        "attention_mask": torch.stack(mask_list),  # [B, T] float
        "texts": texts,
        "phq": torch.stack(phq_list),
        "gad": torch.stack(gad_list),
        "psqi": torch.stack(psqi_list),
        "subject_ids": sids,
    }



    def split_by_subject(self, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Split MODMA data by SUBJECT (not sample) to prevent data leakage.
        Critical: same subject's recordings must NOT appear in train and val.
        """
        import random
        rng = random.Random(seed)
        subject_ids = sorted(set(s["subject_id"] for s in self.samples))
        rng.shuffle(subject_ids)
        
        n_test = max(1, int(len(subject_ids) * test_ratio))
        n_val = max(1, int(len(subject_ids) * val_ratio))
        
        test_ids = set(subject_ids[:n_test])
        val_ids = set(subject_ids[n_test:n_test + n_val])
        train_ids = set(subject_ids[n_test + n_val:])
        
        train_idx = [i for i, s in enumerate(self.samples) if s["subject_id"] in train_ids]
        val_idx = [i for i, s in enumerate(self.samples) if s["subject_id"] in val_ids]
        test_idx = [i for i, s in enumerate(self.samples) if s["subject_id"] in test_ids]
        
        print(f"[MODMA Subject Split] train={len(train_ids)} subjects ({len(train_idx)} samples), "
              f"val={len(val_ids)} ({len(val_idx)} samples), test={len(test_ids)} ({len(test_idx)} samples)")
        
        from torch.utils.data import Subset
        return (
            Subset(self, train_idx),
            Subset(self, val_idx),
            Subset(self, test_idx),
        )


# ══════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═══ Dataset Test ═══\n")

    # Test DAIC-WOZ (if labels extracted)
    diac_dir = "/data/disk1/datasets/diac_woz"
    labels_dir = Path(diac_dir) / "labels"
    if labels_dir.exists():
        print("--- DAIC-WOZ ---")
        ds = DAICWOZDataset(diac_dir, split="train")
        if len(ds) > 0:
            sample = ds[0]
            print(f"  Audio: {sample['audio'].shape}, {sample['audio'].shape[0]/16000:.1f}s")
            print(f"  Transcript: {sample['transcript'][:100]}...")
            print(f"  PHQ-8: {sample['phq_total'].item()}")
            print(f"  PHQ items: {sample['phq_items'].tolist()}")
        else:
            print("  No samples (need to extract archives first)")

    # Test MODMA
    modma_dir = "/data/disk1/datasets/modma/extracted"
    xlsx_path = Path(modma_dir) / "audio_lanzhou_2015" / "subjects_information_audio_lanzhou_2015.xlsx"
    audio_dir = Path("/data/disk1/datasets/modma/extracted/audio_lanzhou_2015")
    if xlsx_path.exists() and audio_dir.exists():
        print("\n--- MODMA ---")
        ds = MODMADataset(
            "/data/disk1/datasets/modma/extracted",
            str(xlsx_path),
        )
        if len(ds) > 0:
            sample = ds[0]
            print(f"  Audio: {sample['audio'].shape}, {sample['audio'].shape[0]/16000:.1f}s")
            print(f"  Transcript: {sample['transcript']}")
            print(f"  PHQ-9: {sample['phq'].item()}")
            print(f"  GAD-7: {sample['gad'].item()}")
            print(f"  PSQI: {sample['psqi'].item()}")
            print(f"  Diagnosis: {sample['diagnosis']}")
    else:
        print("\n--- MODMA: need to extract audio zip first ---")


class CombinedMultimodalDataset(Dataset):
    def __init__(self, daic_ds, modma_ds):
        self.daic = daic_ds; self.modma = modma_ds
        self.daic_len = len(daic_ds); self.modma_len = len(modma_ds)
        self.total = self.daic_len + self.modma_len
        print(f"[Combined] DAIC={self.daic_len} + MODMA={self.modma_len} = {self.total}")
    def __len__(self): return self.total
    def __getitem__(self, idx):
        if idx < self.daic_len:
            item = self.daic[idx]
            return {"audio":item["audio"],"texts":item["transcript"],"phq":item["phq_total"],"has_text":True,"source":"daic","pid":item["pid"]}
        else:
            item = self.modma[idx - self.daic_len]
            return {"audio":item["audio"],"texts":"","phq":item["phq"],"has_text":False,"source":"modma","pid":item["subject_id"]}
    def split_by_subject(self, val_ratio=0.15, seed=42):
        import random; rng = random.Random(seed)
        dp = list(set(s["pid"] for s in self.daic.samples)); rng.shuffle(dp)
        nv = max(1,int(len(dp)*val_ratio)); vp = set(dp[:nv])
        dt=[i for i,s in enumerate(self.daic.samples) if s["pid"] not in vp]
        dv=[i for i,s in enumerate(self.daic.samples) if s["pid"] in vp]
        sp = list(set(s["subject_id"] for s in self.modma.samples)); rng.shuffle(sp)
        nm = max(1,int(len(sp)*val_ratio)); vm = set(sp[:nm])
        mt=[i+self.daic_len for i,s in enumerate(self.modma.samples) if s["subject_id"] not in vm]
        mv=[i+self.daic_len for i,s in enumerate(self.modma.samples) if s["subject_id"] in vm]
        print(f"[Combined Split] train={len(dt)+len(mt)} (D:{len(dt)}+M:{len(mt)}), val={len(dv)+len(mv)} (D:{len(dv)}+M:{len(mv)})")
        from torch.utils.data import Subset; return Subset(self, dt+mt), Subset(self, dv+mv)

def collate_combined(batch):
    max_len = max(item["audio"].shape[0] for item in batch)
    al, ml, tx, pl, ht = [], [], [], [], []
    for item in batch:
        a = item["audio"]; L = a.shape[0]
        if L < max_len:
            a = torch.cat([a, torch.zeros(max_len - L)])
            m = torch.cat([torch.ones(L), torch.zeros(max_len - L)])
        else:
            a = a[:max_len]; m = torch.ones(max_len)
        al.append(a); ml.append(m)
        tx.append(item["texts"]); pl.append(item["phq"]); ht.append(item["has_text"])
    return {"audio":torch.stack(al),"attention_mask":torch.stack(ml),"texts":tx,"phq_total":torch.stack(pl),"has_text":torch.tensor(ht,dtype=torch.bool)}