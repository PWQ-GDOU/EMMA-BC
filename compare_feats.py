import os, sys, json, glob
import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, '/data/disk3/EMMA-BC/phaseB')
from multimodal_dataset import DAICWOZDataset

SYNTH_DIR = '/data/disk3/EMMA-BC/synthetic'
DATA_DIR = '/data/disk1/datasets/diac_woz'

# Load real DAIC-WOZ data
print("Loading DAIC-WOZ...")
ds = DAICWOZDataset(DATA_DIR, split='train', filter_interviewer=False, min_audio_sec=1.0, max_audio_sec=30)

# Get depressed samples (PHQ >= 10)
depressed_samples = [s for s in ds.samples if s.get('phq_total', 0) >= 10]
print("Depressed samples:", len(depressed_samples))

def extract_features(y, sr):
    if len(y) < sr * 1:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    f0, vf, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    f0_v = f0[vf] if np.any(vf) else np.array([100])
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return {
        'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
        'f0_mean': float(np.mean(f0_v)),
        'f0_std': float(np.std(f0_v)),
        'f0_voiced': float(np.mean(vf)),
        'zcr_mean': float(np.mean(zcr)),
        'duration': float(len(y)/sr),
    }

# Extract real
real_feats = []
for s in depressed_samples[:20]:
    path = s['path']
    if not os.path.exists(path):
        continue
    y, sr = sf.read(path)
    if sr != 16000:
        import librosa; y = librosa.resample(y=y, orig_sr=sr, target_sr=16000); sr = 16000
    if len(y) > sr * 30:
        y = y[:sr * 30]
    f = extract_features(y, sr)
    if f:
        real_feats.append(f)

# Extract synthetic
synth_feats = []
for f in sorted(glob.glob(os.path.join(SYNTH_DIR, 'synth_*.wav'))):
    y, sr = sf.read(f)
    fe = extract_features(y, sr)
    if fe:
        synth_feats.append(fe)

print("Real:", len(real_feats), "Synth:", len(synth_feats))

if len(real_feats) > 0 and len(synth_feats) > 0:
    print("
=== COMPARISON ===")
    print("{:<20s} {:>10s} {:>10s}".format("Feature", "Real", "Synth"))
    print("-" * 40)
    keys = ['f0_mean', 'f0_std', 'f0_voiced', 'zcr_mean', 'duration']
    for key in keys:
        r = np.mean([f[key] for f in real_feats])
        s = np.mean([f[key] for f in synth_feats])
        print("{:<20s} {:>10.3f} {:>10.3f}".format(key, r, s))
    
    print("
=== MFCC (first 5 dims) ===")
    for d in range(5):
        r = np.mean([f['mfcc_mean'][d] for f in real_feats])
        s = np.mean([f['mfcc_mean'][d] for f in synth_feats])
        print("MFCC" + str(d+1) + ": real=" + str(round(r,2)) + " synth=" + str(round(s,2)))
