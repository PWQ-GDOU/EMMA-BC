import subprocess, json, os, re, numpy as np
from scipy import stats

SEEDS = [42, 142, 242]
MODES = ["full","audio_only","text_only","video_only","audio_video","video_text","audio_text"]

def run_one(ds, mode, seed):
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    r = subprocess.run(["venv/bin/python", "-u", "train_fusion.py", "--mode", ds, "--modality", mode, "--seed", str(seed)],
                      capture_output=True, text=True, cwd="/data/disk1/CancerEmotionSystem", env=env)
    for line in r.stdout.split(chr(10)):
        if ds.upper() in line and "Acc=" in line:
            acc = float(re.search(r"Acc=([\d.]+)", line).group(1))
            f1 = float(re.search(r"F1=([\d.]+)", line).group(1))
            auc = float(re.search(r"AUC=([\d.]+)", line).group(1))
            return {"acc": acc, "f1": f1, "auc": auc}
    return None

all_results = {}
for ds in ["daic", "modma"]:
    all_results[ds] = {}
    for mode in MODES:
        print(f"{ds} {mode}:", end=" ", flush=True)
        vals = []
        for seed in SEEDS:
            r = run_one(ds, mode, seed)
            if r: vals.append(r)
        if len(vals) >= 2:
            f1s = [v["f1"] for v in vals]
            all_results[ds][mode] = {
                "acc_mean": float(np.mean([v["acc"] for v in vals])),
                "acc_std": float(np.std([v["acc"] for v in vals], ddof=1)),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s, ddof=1)),
                "auc_mean": float(np.mean([v["auc"] for v in vals])),
                "individual": vals, "n_seeds": len(vals),
            }
            print(f"F1={np.mean(f1s):.3f}+-{np.std(f1s,ddof=1):.3f}")
        else:
            print("FAILED")

print("\n=== Statistical Tests ===")
for ds in ["daic", "modma"]:
    res = all_results[ds]
    sorted_modes = sorted(res.keys(), key=lambda m: res[m]["f1_mean"], reverse=True)
    best = sorted_modes[0]; second = sorted_modes[1]
    t, p = stats.ttest_rel([r["f1"] for r in res[best]["individual"]],
                           [r["f1"] for r in res[second]["individual"]])
    sig = "* p<0.05" if p < 0.05 else "n.s."
    print(f"{ds.upper()}: {best}(F1={res[best]['f1_mean']:.3f}) vs {second}(F1={res[second]['f1_mean']:.3f}): t={t:.3f}, p={p:.4f} {sig}")

json.dump(all_results, open("results/multiseed_final.json", "w"), indent=2)
print("\nSaved results/multiseed_final.json")
