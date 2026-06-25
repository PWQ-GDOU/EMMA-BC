#!/usr/bin/env python3
"""
EMMA-BC — Clinical Evaluation Module
════════════════════════════════════
Addresses three clinical deployment gaps:
  1. Binary classification metrics (sensitivity/specificity/F1) at PHQ-8 >= 10
  2. Missing-modality fallback with confidence scoring
  3. Uncertainty quantification via MC-Dropout (95% CI)
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phaseB.multimodal_model import MultimodalClinicalModel, set_seed
from phaseB.multimodal_dataset import DAICWOZDataset, collate_daic
from phaseB.phaseB_train import LabelNormalizer
from torch.utils.data import DataLoader

set_seed(42)

# ═══════════════════════════════════════════════════
# 1. Binary classification metrics (PHQ-8 >= 10)
# ═══════════════════════════════════════════════════

PHQ_CLINICAL_THRESHOLD = 10.0  # Moderate depression cutoff

def classification_metrics(preds, labels):
    """
    Binary metrics for clinical screening.
    PHQ-8 >= 10 = positive (clinically significant depression).
    
    Returns: accuracy, sensitivity, specificity, PPV, NPV, F1, confusion_matrix
    """
    pred_binary = (preds >= PHQ_CLINICAL_THRESHOLD).long()
    label_binary = (labels >= PHQ_CLINICAL_THRESHOLD).long()
    
    tp = ((pred_binary == 1) & (label_binary == 1)).sum().item()
    tn = ((pred_binary == 0) & (label_binary == 0)).sum().item()
    fp = ((pred_binary == 1) & (label_binary == 0)).sum().item()
    fn = ((pred_binary == 0) & (label_binary == 1)).sum().item()
    total = len(preds)
    
    sensitivity = tp / (tp + fn + 1e-8)  # Recall
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / total
    ppv = tp / (tp + fp + 1e-8)  # Precision
    npv = tn / (tn + fn + 1e-8)
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity + 1e-8)
    
    return {
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),  # True positive rate
        "specificity": round(specificity, 4),  # True negative rate
        "ppv": round(ppv, 4),                  # Positive predictive value
        "npv": round(npv, 4),                  # Negative predictive value
        "f1": round(f1, 4),
        "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "threshold": PHQ_CLINICAL_THRESHOLD,
    }


# ═══════════════════════════════════════════════════
# 2. Missing-modality fallback
# ═══════════════════════════════════════════════════

def detect_silence(waveform, threshold_db=-60):
    """Detect if audio is effectively silent."""
    if waveform.numel() == 0:
        return True
    rms = torch.sqrt(torch.mean(waveform ** 2))
    db = 20 * torch.log10(rms + 1e-10)
    return db.item() < threshold_db


def detect_empty_text(text):
    """Detect empty or whitespace-only transcript."""
    return text is None or len(text.strip()) < 10


@torch.no_grad()
    """
    Missing-modality fallback prediction.
    
    WARNING: Audio-only and text-only branches were NOT trained on
    single-modality data. Fallback predictions may be unreliable.
    For paper evaluation, only use complete dual-modality samples.
    Single-modality fallback is a demo placeholder for future work.
    """
def predict_with_fallback(model, audio, transcript, attention_mask, device, 
                           confidence_threshold=0.5):
    """
    Predict PHQ-8 with missing-modality fallback.
    
    - If audio is silent: use text-only prediction, flag low confidence
    - If text is empty: use audio-only prediction, flag low confidence
    - If both available: full multimodal prediction, high confidence
    
    Returns: pred, confidence_score, modality_flag
    """
    audio_missing = detect_silence(audio.squeeze(0)) if audio.numel() > 0 else True
    text_missing = detect_empty_text(transcript)
    
    if audio_missing and text_missing:
        # Both missing — return dataset mean as fallback
        return 10.0, 0.0, "both_missing"
    
    if audio_missing:
        # Text-only: bypass audio encoder, feed zero audio embedding
        text_emb = model.text_encoder([transcript], device)
        dummy_audio = torch.zeros(1, model.d_model, device=device)
        outputs = model.fusion_head(dummy_audio, text_emb)
        conf = 0.3  # Low confidence: missing modality
        flag = "audio_missing"
    elif text_missing:
        # Audio-only: bypass text encoder
        audio_emb = model.audio_encoder(audio, attention_mask)
        dummy_text = torch.zeros(1, model.d_model, device=device)
        outputs = model.fusion_head(audio_emb, dummy_text)
        conf = 0.3  # Low confidence: missing modality
        flag = "text_missing"
    else:
        # Full multimodal
        outputs = model(audio, [transcript], attention_mask)
        conf = 0.9  # High confidence: both modalities present
        flag = "full"
    
    pred = outputs["phq"].squeeze(-1).cpu().item()
    return pred, conf, flag


# ═══════════════════════════════════════════════════
# 3. Uncertainty quantification via MC-Dropout
# ═══════════════════════════════════════════════════

def enable_dropout(model):
    """
    MC-Dropout: re-enable Dropout while keeping BN/LN frozen.
    model.eval() freezes ALL modules; this selectively re-enables Dropout.
    BN/LN stay frozen to prevent stat drift during MC sampling.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            module.eval()


@torch.no_grad()
def mc_dropout_predict(model, audio, texts, attention_mask, device, n_samples=30):
    """
    Monte Carlo Dropout: run forward pass N times with dropout enabled.
    
    Returns: mean_pred, std_pred, ci_lower (percentile), ci_upper (percentile), all_preds
    """
    model.eval()
    enable_dropout(model)
    
    all_preds = []
    for _ in range(n_samples):
        outputs = model(audio, texts, attention_mask)
        preds = outputs["phq"].squeeze(-1).cpu()  # [B]
        all_preds.append(preds)
    
    all_preds = torch.stack(all_preds, dim=0)  # [N, B]
    mean_preds = all_preds.mean(dim=0)          # [B]
    std_preds = all_preds.std(dim=0)            # [B]
    # Percentile-based CI: PHQ-8 is bounded [0,24], Gaussian assumption is wrong
    lower_95 = torch.tensor(np.percentile(all_preds.numpy(), 2.5, axis=0))
    upper_95 = torch.tensor(np.percentile(all_preds.numpy(), 97.5, axis=0))
    
    return {
        "mean": mean_preds,
        "std": std_preds,
        "ci_lower": lower_95,
        "ci_upper": upper_95,
        "all_samples": all_preds,
    }


def mc_dropout_eval(model, loader, normalizer, device, n_samples=30):
    """Run MC-Dropout evaluation on full dataset."""
    model.eval()
    enable_dropout(model)
    
    all_means, all_stds, all_lowers, all_uppers = [], [], [], []
    all_labels = []
    
    print(f"Running MC-Dropout ({n_samples} samples)...")
    for batch in tqdm(loader):
        audio = batch["audio"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        texts = batch["texts"]
        labels = batch["phq_total"]
        
        if torch.isnan(labels).all():
            continue
        if attention_mask.sum() < 1:
            continue
        
        result = mc_dropout_predict(model, audio, texts, attention_mask, 
                                      device, n_samples=n_samples)
        
        norm_mean = normalizer.mean[0].item() if normalizer.mean is not None else 0
        norm_std = normalizer.std[0].item() if normalizer.std is not None else 1
        
        all_means.append(result["mean"] * norm_std + norm_mean)
        all_stds.append(result["std"] * norm_std)  # Std scales with normalization
        all_lowers.append(result["ci_lower"] * norm_std + norm_mean)
        all_uppers.append(result["ci_upper"] * norm_std + norm_mean)
        all_labels.append(labels)
    
    means = torch.cat(all_means)
    stds = torch.cat(all_stds)
    lowers = torch.cat(all_lowers)
    uppers = torch.cat(all_uppers)
    labels = torch.cat(all_labels)
    
    # Check coverage: what fraction of true labels fall within 95% CI?
    in_ci = ((labels >= lowers) & (labels <= uppers)).float().mean().item()
    
    return {
        "pred_mean": means.tolist(),
        "pred_std": stds.tolist(),
        "ci_lower": lowers.tolist(),
        "ci_upper": uppers.tolist(),
        "labels": labels.tolist(),
        "coverage_95": round(in_ci, 4),
        "mean_width": round((uppers - lowers).mean().item(), 2),
        "n_samples": n_samples,
    }


# ═══════════════════════════════════════════════════
# Full clinical evaluation
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="/data/disk1/datasets/diac_woz")
    parser.add_argument("--output", type=str, default="clinical_results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mc_samples", type=int, default=30)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    print("Loading model...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MultimodalClinicalModel(
        d_model=256, n_layers=4, n_heads=8,
        freeze_audio_w2v=True, freeze_text_bert=True, n_tasks=1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    normalizer = LabelNormalizer()
    if "label_normalizer" in ckpt:
        normalizer.load_state_dict(ckpt["label_normalizer"])
        normalizer.mean = normalizer.mean.to(device)
        normalizer.std = normalizer.std.to(device)
    
    # Data
    print("Loading validation data...")
    val_ds = DAICWOZDataset(args.data, split="train",
                             filter_interviewer=False,  # Use full transcript for eval
                             min_audio_sec=1.0, max_audio_sec=600)
    _, val_ds = val_ds.split_val_from_train(val_ratio=0.15, seed=42)
    
    gen = torch.Generator().manual_seed(42)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_daic, num_workers=2, pin_memory=True,
        generator=gen,
    )
    print(f"Validation: {len(val_ds)} samples")
    
    # ═══ 1. Classification metrics ═══
    print("\n═══ Clinical Binary Classification ═══")
    model.eval()
    all_preds_cls, all_labels_cls = [], []
    
    for batch in tqdm(val_loader, desc="Classify"):
        audio = batch["audio"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        texts = batch["texts"]
        labels = batch["phq_total"]
        
        if torch.isnan(labels).all():
            continue
        if attention_mask.sum() < 1:
            continue
        
        outputs = model(audio, texts, attention_mask=attention_mask)
        preds_norm = outputs["phq"].squeeze(-1).cpu()
        denorm_preds = preds_norm * normalizer.std[0].cpu() + normalizer.mean[0].cpu()
        
        all_preds_cls.append(denorm_preds)
        all_labels_cls.append(labels)
    
    preds = torch.cat(all_preds_cls)
    labels = torch.cat(all_labels_cls)
    
    cls_metrics = classification_metrics(preds, labels)
    print(f"Threshold: PHQ-8 >= {cls_metrics['threshold']}")
    print(f"Accuracy:   {cls_metrics['accuracy']:.3f}")
    print(f"Sensitivity:{cls_metrics['sensitivity']:.3f} (how many true patients caught)")
    print(f"Specificity:{cls_metrics['specificity']:.3f} (how many healthy correctly ruled out)")
    print(f"F1:         {cls_metrics['f1']:.3f}")
    print(f"Confusion:  TP={cls_metrics['confusion']['TP']}, TN={cls_metrics['confusion']['TN']}, "
          f"FP={cls_metrics['confusion']['FP']}, FN={cls_metrics['confusion']['FN']}")
    
    with open(os.path.join(args.output, "classification.json"), "w") as f:
        json.dump(cls_metrics, f, indent=2)
    
    # ═══ 2. Missing-modality fallback demo ═══
    print("\n═══ Missing-Modality Fallback ═══")
    # Test on first 3 samples
    for idx in range(min(3, len(val_ds))):
        sample = val_ds[idx]
        audio = sample["audio"].unsqueeze(0).to(device)
        mask = torch.ones(1, audio.shape[1], device=device)
        text = sample["transcript"]
        
        # Normal prediction
        pred_full, conf_full, flag_full = predict_with_fallback(
            model, audio, text, mask, device)
        
        # Simulate audio missing
        silent = torch.zeros(1, audio.shape[1], device=device)
        pred_silent, conf_silent, flag_silent = predict_with_fallback(
            model, silent, text, mask, device)
        
        # Simulate text missing
        pred_notext, conf_notext, flag_notext = predict_with_fallback(
            model, audio, "", mask, device)
        
        print(f"  PID={sample['pid']}")
        print(f"    Full:     PHQ={pred_full:.1f} conf={conf_full:.1f} flag={flag_full}")
        print(f"    No audio: PHQ={pred_silent:.1f} conf={conf_silent:.1f} flag={flag_silent}")
        print(f"    No text:  PHQ={pred_notext:.1f} conf={conf_notext:.1f} flag={flag_notext}")
    
    # ═══ 3. Uncertainty quantification ═══
    print(f"\n═══ MC-Dropout Uncertainty ({args.mc_samples} samples) ═══")
    mc_results = mc_dropout_eval(model, val_loader, normalizer, device, 
                                   n_samples=args.mc_samples)
    
    print(f"95% CI coverage: {mc_results['coverage_95']:.3f} "
          f"(target: >= 0.90 for well-calibrated model)")
    print(f"Mean 95% CI width: {mc_results['mean_width']:.1f} PHQ-8 points")
    
    # Show example uncertainties
    print(f"\nExample predictions with uncertainty:")
    for i in range(min(5, len(mc_results['pred_mean']))):
        pred = mc_results['pred_mean'][i]
        lo = mc_results['ci_lower'][i]
        hi = mc_results['ci_upper'][i]
        true = mc_results['labels'][i]
        in_ci = "IN" if lo <= true <= hi else "OUT"
        print(f"  [{in_ci}] True={true:<5.1f} Pred={pred:<5.1f} 95%CI=[{lo:.1f}, {hi:.1f}]")
    
    # Save (only summary stats, not all samples — too large)
    summary = {k: v for k, v in mc_results.items() 
               if k not in ("pred_mean", "pred_std", "ci_lower", "ci_upper", "labels")}
    with open(os.path.join(args.output, "uncertainty.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {args.output}/")


def predict_to_csv(model, loader, normalizer, device, output_path,
                    use_mc_dropout=True, mc_samples=30):
    """Generate prediction CSV with Fused_Trust_Score.
    Columns: Subject_ID, PHQ_Score, Confidence, Modality_Status,
    MC_Mean, MC_Std, CI_Lower_95, CI_Upper_95, Fused_Trust_Score"""
    import csv
    rows = []
    max_mc_std = 0.0
    model.eval()
    for batch in tqdm(loader, desc="Predict CSV"):
        audio = batch["audio"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        texts = batch["texts"]
        pids = batch["pids"]
        labels = batch["phq_total"]
        if torch.isnan(labels).all() or attention_mask.sum() < 1:
            continue
        for i in range(len(pids)):
            a = audio[i:i+1]; m = attention_mask[i:i+1]; t = texts[i]; pid = pids[i]
            pred, conf, flag = predict_with_fallback(model, a, t, m, device)
            if use_mc_dropout:
                mc_result = mc_dropout_predict(model, a, [t], m, device, n_samples=mc_samples)
                nm = normalizer.mean[0].item() if normalizer.mean is not None else 0
                ns = normalizer.std[0].item() if normalizer.std is not None else 1
                mc_mean = mc_result["mean"][0].item() * ns + nm
                mc_std = mc_result["std"][0].item() * ns
                ci_lo = mc_result["ci_lower"][0].item() * ns + nm
                ci_hi = mc_result["ci_upper"][0].item() * ns + nm
                max_mc_std = max(max_mc_std, mc_std)
                rows.append([pid, pred, conf, flag, mc_mean, mc_std, ci_lo, ci_hi, 0.0])
            else:
                rows.append([pid, round(pred,2), round(conf,2), flag])
    if use_mc_dropout and max_mc_std > 0:
        for row in rows:
            row[8] = round(row[2] * (1.0 - row[5] / max_mc_std), 2)
    header = ["Subject_ID","PHQ_Score","Confidence","Modality_Status",
              "MC_Mean","MC_Std","CI_Lower_95","CI_Upper_95","Fused_Trust_Score"]
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Predictions saved to {output_path} ({len(rows)} records)")
    return output_path



    main()
