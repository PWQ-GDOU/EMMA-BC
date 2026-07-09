#!/usr/bin/env python3
"""Multi-seed robustness: run Phase B ablation with 5 seeds, aggregate mean±std."""
import subprocess, json, os, sys, argparse
from pathlib import Path
import numpy as np
from scipy import stats

BASE = Path('/data/disk1/CancerEmotionSystem')
VENV = BASE / 'venv/bin/python'
SEEDS = [42, 142, 242, 342, 442]
MODALITIES = ['full', 'audio_only', 'text_only', 'video_only', 'audio_video', 'video_text', 'all']
GPU = 'cuda:0'
EPOCHS = 50
BATCH = 2

def run_one(modality, seed, ckpt_dir):
    cmd = [
        str(VENV), str(BASE/'phaseB/phaseB_train.py'),
        '--device', GPU, '--epochs', str(EPOCHS), '--batch_size', str(BATCH),
        '--modality', modality, '--seed', str(seed),
        '--audio_pretrained', str(BASE/'checkpoints/phaseA/phaseA_best.pt'),
        '--num_workers', '0', '--checkpoint_dir', ckpt_dir,
    ]
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'PYTHONUNBUFFERED': '1'}
    log = BASE / f'logs/multiseed_{modality}_seed{seed}.log'
    with open(log, 'w') as f:
        p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f'  FAILED (seed={seed}): check {log}')
        return None
    # Parse best MAE from log
    text = open(log).read()
    for line in text.split('\n'):
        if 'Best MAE:' in line:
            parts = line.split(',')
            mae = float(parts[0].split(':')[1].strip())
            ccc = float(parts[2].split(':')[1].strip()) if len(parts) > 2 else 0.0
            return {'mae': mae, 'ccc': ccc, 'seed': seed}
    print(f'  No Best MAE found in {log}')
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modalities', nargs='+', default=['full','all','video_only'],
                       help='Which modalities to run')
    parser.add_argument('--gpu', default='cuda:1')
    args = parser.parse_args()
    global GPU; GPU = args.gpu
    
    results = {}
    for mod in args.modalities:
        print(f'\n=== {mod} ===')
        mod_results = []
        for seed in SEEDS:
            ckpt_dir = str(BASE / f'checkpoints/multiseed_{mod}_s{seed}')
            print(f'  seed={seed}...', end=' ', flush=True)
            r = run_one(mod, seed, ckpt_dir)
            if r: 
                mod_results.append(r)
                print(f'MAE={r["mae"]:.3f} CCC={r["ccc"]:.3f}')
        if len(mod_results) >= 2:
            maes = [r['mae'] for r in mod_results]
            cccs = [r['ccc'] for r in mod_results]
            results[mod] = {
                'mae_mean': float(np.mean(maes)), 'mae_std': float(np.std(maes, ddof=1)),
                'ccc_mean': float(np.mean(cccs)), 'ccc_std': float(np.std(cccs, ddof=1)),
                'individual': mod_results, 'n_runs': len(mod_results),
            }
            print(f'  => MAE={np.mean(maes):.3f}±{np.std(maes,ddof=1):.3f}  CCC={np.mean(cccs):.3f}±{np.std(cccs,ddof=1):.3f}')
    
    # Paired t-test between best two
    if 'full' in results and 'all' in results:
        a = [r['mae'] for r in results['full']['individual']]
        b = [r['mae'] for r in results['all']['individual']]
        t, p = stats.ttest_rel(a, b)
        results['ttest_full_vs_all'] = {'t': float(t), 'p': float(p), 'significant': bool(p < 0.05)}
        print(f'\nPaired t-test (full vs all): t={t:.3f} p={p:.4f} {"SIGNIFICANT" if p<0.05 else "n.s."}')
    
    out = BASE / 'results/multiseed_robustness.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {out}')

if __name__ == '__main__':
    main()
