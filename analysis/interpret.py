#!/usr/bin/env python3
"""
EMMA-BC — Model Interpretability & Analysis Module
══════════════════════════════════════════════════
Provides:
  1. Feature permutation importance (modal contribution)
  2. Error case analysis (worst predictions)
  3. Interviewer prompt filtering for DAIC-WOZ transcripts
  4. Subgroup fairness analysis (gender, age)

Usage:
  python analysis/interpret.py \\
    --checkpoint checkpoints/phaseB/phaseB_best.pt \\
    --data /data/disk1/datasets/diac_woz \\
    --output analysis_results/
"""

import os, sys, json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phaseB.multimodal_model import MultimodalClinicalModel, regression_metrics, set_seed
from phaseB.multimodal_dataset import DAICWOZDataset, collate_daic
from torch.utils.data import DataLoader

set_seed(42)

# ═══════════════════════════════════════════════════
# 1. DAIC-WOZ Interviewer prompt filter
# ═══════════════════════════════════════════════════

# Common Ellie (virtual interviewer) prompt patterns in DAIC-WOZ
ELLIE_PATTERNS = [
    "how are you doing today",
    "what's been going on",
    "tell me about",
    "how do you feel",
    "can you tell me",
    "i'd like to ask",
    "let's talk about",
    "could you describe",
    "what brings you here",
    "how has your mood",
    "on a scale of",
    "you mentioned",
    "can you elaborate",
    "that's interesting",
    "thank you for sharing",
    "i understand",
    "how does that make you feel",
    "would you say",
    "is there anything else",
    "so you're saying",
    "i'm going to ask",
    "tell me more",
    "how long have you",
    "when did you",
]

def filter_interviewer_text(transcript_path, keep_threshold=0.0):
    """
    Filter Ellie (interviewer) prompts from DAIC-WOZ transcript.
    
    DAIC-WOZ transcripts have no speaker labels — Ellie's questions and 
    participant's answers are interleaved as continuous text.
    
    Strategy: Remove segments matching known interviewer patterns.
    Returns participant-only text.
    """
    if transcript_path is None:
        return ""
    
    df = pd.read_csv(transcript_path)
    texts = df["Text"].dropna().tolist()
    
    filtered = []
    for text in texts:
        text_lower = text.lower().strip()
        is_interviewer = any(pattern in text_lower for pattern in ELLIE_PATTERNS)
        if not is_interviewer:
            filtered.append(text)
    
    return " ".join(filtered)


# ═══════════════════════════════════════════════════
# 2. Feature permutation importance
# ═══════════════════════════════════════════════════

@torch.no_grad()
def permutation_importance(model, loader, normalizer, device, n_permutations=5):
    """
    Compute feature importance by permuting each modality separately.
    
    Measures how much MAE increases when each modality input is destroyed.
    Higher MAE increase = more important modality.
    """
    import random
    
    # Baseline: unpermuted
    base_preds, base_labels = get_predictions(model, loader, normalizer, device)
    base_mae = torch.abs(base_preds - base_labels).mean().item()
    
    results = {}
    
    # Audio modality importance
    audio_maes = []
    for _ in range(n_permutations):
        preds, labels = get_predictions(model, loader, normalizer, device,
                                         permute_modality="audio")
        audio_maes.append(torch.abs(preds - labels).mean().item())
    results["audio"] = {"mae": np.mean(audio_maes), "delta": np.mean(audio_maes) - base_mae}
    
    # Text modality importance
    text_maes = []
    for _ in range(n_permutations):
        preds, labels = get_predictions(model, loader, normalizer, device,
                                         permute_modality="text")
        text_maes.append(torch.abs(preds - labels).mean().item())
    results["text"] = {"mae": np.mean(text_maes), "delta": np.mean(text_maes) - base_mae}
    
    # Both permuted (random baseline)
    both_maes = []
    for _ in range(n_permutations):
        preds, labels = get_predictions(model, loader, normalizer, device,
                                         permute_modality="both")
        both_maes.append(torch.abs(preds - labels).mean().item())
    results["both"] = {"mae": np.mean(both_maes), "delta": np.mean(both_maes) - base_mae}
    
    results["baseline_mae"] = base_mae
    return results


@torch.no_grad()
def get_predictions(model, loader, normalizer, device, permute_modality=None):
    """Get model predictions, optionally permuting a modality."""
    all_preds = []
    all_labels = []
    
    for batch in loader:
        audio = batch["audio"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        texts = batch["texts"]
        labels = batch["phq_total"].to(device)
        
        if torch.isnan(labels).all():
            continue
        if attention_mask.sum() < 1:
            continue
        
        # Permute: shuffle inputs to destroy modality-specific information
        if permute_modality == "audio":
            idx = torch.randperm(audio.shape[0])
            audio = audio[idx]
        elif permute_modality == "text":
            idx = torch.randperm(len(texts))
            texts = [texts[i] for i in idx]
        elif permute_modality == "both":
            idx_a = torch.randperm(audio.shape[0])
            audio = audio[idx_a]
            idx_t = torch.randperm(len(texts))
            texts = [texts[i] for i in idx_t]
        
        outputs = model(audio, texts, attention_mask=attention_mask)
        preds = outputs["phq"].squeeze(-1)
        
        all_preds.append(preds)
        all_labels.append(labels)
    
    return torch.cat(all_preds), torch.cat(all_labels)


# ═══════════════════════════════════════════════════
# 3. Error case analysis
# ═══════════════════════════════════════════════════

@torch.no_grad()
def error_analysis(model, full_dataset, normalizer, device, top_k=10):
    """
    Find worst predictions and analyze error patterns.
    
    Returns: list of {pid, true_label, pred_label, abs_error, transcript_preview}
    """
    model.eval()
    
    errors = []
    norm_mean = normalizer.mean.to(device) if normalizer.mean is not None else 0
    norm_std = normalizer.std.to(device) if normalizer.std is not None else 1
    
    for idx in range(len(full_dataset)):
        sample = full_dataset[idx]
        audio = sample["audio"].unsqueeze(0).to(device)
        transcript = sample["transcript"]
        true_label = sample["phq_total"].item()
        
        # Create attention mask
        attention_mask = torch.ones_like(audio).to(device)
        if attention_mask.sum() < 1:
            continue
        
        outputs = model(audio, [transcript], attention_mask=attention_mask)
        pred_norm = outputs["phq"].squeeze(-1).item()
        pred_raw = pred_norm * norm_std[0].item() + norm_mean[0].item()
        
        abs_error = abs(pred_raw - true_label)
        errors.append({
            "pid": sample["pid"] if "pid" in sample else idx,
            "true": true_label,
            "pred": round(pred_raw, 2),
            "error": round(abs_error, 2),
            "transcript": transcript[:200],
        })
    
    errors.sort(key=lambda x: x["error"], reverse=True)
    return errors[:top_k]


# ═══════════════════════════════════════════════════
# 4. Subgroup fairness analysis
# ═══════════════════════════════════════════════════

@torch.no_grad()
def subgroup_analysis(model, loader, normalizer, device, metadata_csv):
    """
    Report performance stratified by gender.
    DAIC-WOZ metadata: Participant_ID, Gender columns.
    """
    if not os.path.exists(metadata_csv):
        return {"error": "metadata_mapped.csv not found"}
    
    meta = pd.read_csv(metadata_csv)
    gender_map = dict(zip(meta["Participant_ID"], meta["Gender"]))
    
    # Actually, use PHQ_Binary from labels
    results = defaultdict(lambda: {"preds": [], "labels": []})
    
    for batch in loader:
        audio = batch["audio"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        texts = batch["texts"]
        labels = batch["phq_total"].to(device)
        pids = batch["pids"]
        
        if torch.isnan(labels).all():
            continue
        if attention_mask.sum() < 1:
            continue
        
        outputs = model(audio, texts, attention_mask=attention_mask)
        preds = outputs["phq"].squeeze(-1)
        
        for pid, pred, label in zip(pids, preds.cpu(), labels.cpu()):
            gender = gender_map.get(pid, "unknown")
            # Binary: depressed (PHQ >= 10) vs not
            subgroup = f"{gender}_dep" if label >= 10 else f"{gender}_ndep"
            results[subgroup]["preds"].append(pred.item())
            results[subgroup]["labels"].append(label.item())
    
    summary = {}
    for group, data in results.items():
        if len(data["preds"]) < 3:
            continue
        preds = torch.tensor(data["preds"])
        labels = torch.tensor(data["labels"])
        denorm_preds = preds * normalizer.std[0] + normalizer.mean[0]
        mae = torch.abs(denorm_preds - labels).mean().item()
        summary[group] = {"n": len(data["preds"]), "mae": round(mae, 2)}
    
    return summary


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="/data/disk1/datasets/diac_woz")
    parser.add_argument("--output", type=str, default="analysis_results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("Loading model...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MultimodalClinicalModel(
        d_model=256, n_layers=4, n_heads=8,
        freeze_audio_w2v=True, freeze_text_bert=True, n_tasks=1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Load normalizer
    from phaseB.phaseB_train import LabelNormalizer
    normalizer = LabelNormalizer()
    if "label_normalizer" in ckpt:
        normalizer.load_state_dict(ckpt["label_normalizer"])
        normalizer.mean = normalizer.mean.to(device)
        normalizer.std = normalizer.std.to(device)
    
    # Data
    print("Loading validation data...")
    full_ds = DAICWOZDataset(args.data, split="train",
                              min_audio_sec=1.0, max_audio_sec=600)
    train_ds, val_ds = full_ds.split_val_from_train(val_ratio=0.15, seed=42)
    
    gen = torch.Generator().manual_seed(42)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_daic, num_workers=2, pin_memory=True,
        generator=gen,
    )
    
    os.makedirs(args.output, exist_ok=True)
    
    # ── 1. Permutation importance ──
    print("\n═══ Permutation Importance ═══")
    importance = permutation_importance(model, val_loader, normalizer, device)
    
    print(f"Baseline MAE: {importance['baseline_mae']:.3f}")
    for mod in ["audio", "text", "both"]:
        r = importance[mod]
        print(f"  {mod}: MAE={r['mae']:.3f}  (delta={r['delta']:+.3f})")
    
    with open(os.path.join(args.output, "importance.json"), "w") as f:
        json.dump(importance, f, indent=2)
    
    # ── 2. Error analysis ──
    print(f"\n═══ Top {args.top_k} Error Cases ═══")
    errors = error_analysis(model, val_ds, normalizer, device, top_k=args.top_k)
    for i, e in enumerate(errors):
        print(f"  {i+1}. PID={e['pid']} | True={e['true']} | Pred={e['pred']} | Err={e['error']}")
        print(f"     Text: {e['transcript'][:100]}...")
    
    with open(os.path.join(args.output, "error_cases.json"), "w") as f:
        json.dump(errors, f, indent=2)
    
    # ── 3. Interviewer filtering demo ──
    print("\n═══ Interviewer Filtering Demo ═══")
    # Pick a random sample and show before/after
    sample = val_ds[0]
    raw = sample["transcript"]
    # Actually the dataset stores raw transcript; show filter effect
    filtered = filter_interviewer_text(
        os.path.join(args.data, "extracted", sample["pid"], f"{sample['pid']}_Transcript.csv")
    )
    print(f"  PID={sample['pid']}")
    print(f"  Raw chars: {len(raw)}, Filtered chars: {len(filtered)}")
    print(f"  Removal: {(1 - len(filtered)/max(1,len(raw)))*100:.0f}%")
    
    # ── 4. Subgroup analysis ──
    print("\n═══ Subgroup Analysis ═══")
    meta_csv = os.path.join(args.data, "metadata_mapped.csv")
    subgroups = subgroup_analysis(model, val_loader, normalizer, device, meta_csv)
    for group, info in sorted(subgroups.items()):
        print(f"  {group}: n={info['n']}, MAE={info['mae']}")
    with open(os.path.join(args.output, "subgroup_analysis.json"), "w") as f:
        json.dump(subgroups, f, indent=2)
    
    print(f"\nResults saved to {args.output}/")
    

if __name__ == "__main__":
    main()
