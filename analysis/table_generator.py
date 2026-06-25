#!/usr/bin/env python3
"""
table_generator.py — Paper-ready table materials for EMMA-BC
Usage: python analysis/table_generator.py --csv checkpoints/predictions.csv
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_hline(title=""):
    w = max(60, len(title) + 4)
    print(f"\n{'─'*w}")
    if title:
        print(f"  {title}")
        print(f"{'─'*w}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Prediction CSV from predict_to_csv")
    parser.add_argument("--labels", default=None, help="Optional: path to DAIC-WOZ labels CSV for gender/PHQ true labels")
    parser.add_argument("--output", default=None, help="Save as JSON (optional)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Sanity checks
    required = ["PHQ_Score", "Modality_Status", "Binary_Label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return

    n = len(df)
    n_full = (df["Modality_Status"] == "Full").sum()
    print(f"Total samples: {n}  (Full modality: {n_full}, "
          f"Degraded: {n - n_full})")

    # ═══════════════════════════════════════════
    # TABLE 1: Overall Regression Performance
    # ═══════════════════════════════════════════
    print_hline("TABLE 1 — Regression Performance")

    phq_pred = df["PHQ_Score"].values
    mae = None
    rmse = None
    if "MC_Mean" in df.columns:
        mc_mean = df["MC_Mean"].values
        mc_std = df["MC_Std"].values
        mean_ci_width = np.mean(2 * mc_std)
        print(f"  MAE   (point):        —")
        print(f"  RMSE  (MC mean):       —")
        print(f"  Mean 95% CI width:    {mean_ci_width:.2f}  PHQ-8 points")
    else:
        print(f"  (MC-Dropout columns not found in CSV)")

    # If true labels available
    if args.labels and os.path.exists(args.labels):
        labels_df = pd.read_csv(args.labels)
        labels_df.set_index("Participant_ID", inplace=True)
        # Match with predictions
        true_labels = []
        pred_labels = []
        for _, row in df.iterrows():
            pid = str(row.get("Subject_ID", ""))
            if pid in labels_df.index:
                try:
                    true_labels.append(float(labels_df.loc[pid, "PHQ_8Total"]))
                    pred_labels.append(float(row["PHQ_Score"]))
                except (ValueError, KeyError):
                    pass
        if true_labels:
            true_a = np.array(true_labels)
            pred_a = np.array(pred_labels)
            mae = np.mean(np.abs(pred_a - true_a))
            rmse = np.sqrt(np.mean((pred_a - true_a) ** 2))
            ccc_val = np.corrcoef(true_a, pred_a)[0, 1]
            print(f"\n  MAE:    {mae:.3f}")
            print(f"  RMSE:   {rmse:.3f}")
            print(f"  Pearson r: {ccc_val:.3f}")
            print(f"  N (matched): {len(true_a)}")

    # ═══════════════════════════════════════════
    # TABLE 2: Binary Classification
    # ═══════════════════════════════════════════
    print_hline("TABLE 2 — Binary Classification (PHQ-8 >= 10)")

    n_dep = (df["Binary_Label"] == "DEPRESSED").sum()
    n_ndep = (df["Binary_Label"] == "NON-DEPRESSED").sum()
    print(f"  Predicted DEPRESSED:    {n_dep}  ({100*n_dep/n:.1f}%)")
    print(f"  Predicted NON-DEPRESSED:{n_ndep}  ({100*n_ndep/n:.1f}%)")

    if args.labels and os.path.exists(args.labels) and true_labels:
        true_bin = (true_a >= 10)
        pred_bin = (pred_a >= 10)
        tp = ((pred_bin) & (true_bin)).sum()
        tn = ((~pred_bin) & (~true_bin)).sum()
        fp = ((pred_bin) & (~true_bin)).sum()
        fn = ((~pred_bin) & (true_bin)).sum()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        ppv = tp / (tp + fp + 1e-8)
        acc = (tp + tn) / len(true_bin)
        f1 = 2 * ppv * sens / (ppv + sens + 1e-8)
        print(f"\n  Accuracy:     {acc:.3f}")
        print(f"  Sensitivity:  {sens:.3f}")
        print(f"  Specificity:  {spec:.3f}")
        print(f"  PPV:          {ppv:.3f}")
        print(f"  F1:           {f1:.3f}")
        print(f"  Confusion:    TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    # ═══════════════════════════════════════════
    # TABLE 3: Trust Score Analysis
    # ═══════════════════════════════════════════
    if "Fused_Trust_Score" in df.columns:
        print_hline("TABLE 3 — Trust Score Distribution")
        ts = df["Fused_Trust_Score"]
        print(f"  Mean:   {ts.mean():.3f}")
        print(f"  Median: {ts.median():.3f}")
        print(f"  Std:    {ts.std():.3f}")
        print(f"  Min:    {ts.min():.3f}")
        print(f"  Max:    {ts.max():.3f}")
        print(f"  Q25/Q75: {ts.quantile(0.25):.3f} / {ts.quantile(0.75):.3f}")
        low_trust = (ts < 0.3).sum()
        print(f"  Low-trust (<0.3): {low_trust} ({100*low_trust/n:.1f}%)")

    # ═══════════════════════════════════════════
    # TABLE 4: Modality Status Breakdown
    # ═══════════════════════════════════════════
    print_hline("TABLE 4 — By Modality Status")
    for status in sorted(df["Modality_Status"].unique()):
        subset = df[df["Modality_Status"] == status]
        m = len(subset)
        phq_mean = subset["PHQ_Score"].mean()
        ts_mean = subset["Fused_Trust_Score"].mean() if "Fused_Trust_Score" in df.columns else float('nan')
        print(f"  {status:<20s}  N={m:<4d}  PHQ_Mean={phq_mean:.2f}  Trust={ts_mean:.3f}")

    # ═══════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════
    if args.output:
        report = {
            "n_total": n, "n_full_modality": int(n_full),
            "mae": float(mae) if mae else None,
            "rmse": float(rmse) if rmse else None,
            "sensitivity": float(sens) if 'sens' in dir() else None,
            "specificity": float(spec) if 'spec' in dir() else None,
            "f1": float(f1) if 'f1' in dir() else None,
            "trust_score": {k: float(v) for k, v in ts.describe().to_dict().items()} if "Fused_Trust_Score" in df.columns else {},
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
