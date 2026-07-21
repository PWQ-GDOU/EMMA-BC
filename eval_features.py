import os, json, glob, random
import numpy as np
import soundfile as sf
import librosa

SYNTH_DIR = '/data/disk3/EMMA-BC/synthetic'
REAL_DIR = '/data/disk1/datasets/diac_woz'

def extract_features(y, sr):
    """Extract depressive speech features."""
    if len(y) < sr * 2:
        return None
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Pitch (F0)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    f0_valid = f0[voiced_flag] if np.any(voiced_flag) else np.array([0])
    
    # Speech rate (approximated by zero-crossing rate)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    # Spectral centroid
    spectral = np.abs(librosa.stft(y))
    centroid = librosa.feature.spectral_centroid(S=spectral, sr=sr)[0]
    
    return {
        'mfcc_mean': mfcc_mean.tolist(),
        'mfcc_std': mfcc_std.tolist(),
        'f0_mean': float(np.mean(f0_valid)),
        'f0_std': float(np.std(f0_valid)),
        'f0_pct_voiced': float(np.mean(voiced_flag)),
        'zcr_mean': float(np.mean(zcr)),
        'zcr_std': float(np.std(zcr)),
        'centroid_mean': float(np.mean(centroid)),
        'centroid_std': float(np.std(centroid)),
        'duration': float(len(y)/sr),
    }

# Extract synthetic features
print("=== SYNTHETIC ===")
synth_features = []
for f in sorted(glob.glob(os.path.join(SYNTH_DIR, 'synth_*.wav'))):
    y, sr = sf.read(f)
    feats = extract_features(y, sr)
    if feats:
        synth_features.append(feats)

# Extract real depressed features (PHQ >= 10)
print("=== REAL DEPRESSED ===")
real_features = []
# Find real depressed samples - use a few random ones from DAIC-WOZ
real_wavs = []
for root, dirs, files in os.walk(REAL_DIR):
    for f in files:
        if f.endswith('_AUDIO.wav') and '_P/' in root:
            real_wavs.append(os.path.join(root, f))
            if len(real_wavs) >= 50:
                break
    if len(real_wavs) >= 50:
        break

# Take the first 20 real samples as reference (we don't know PHQ labels here, just use them all)
for f in real_wavs[:20]:
    y, sr = sf.read(f, samplerate=16000)
    if len(y) > sr * 30:
        y = y[:sr * 30]
    feats = extract_features(y, sr)
    if feats:
        real_features.append(feats)

# Compare distributions
print("")
print("=== FEATURE DISTRIBUTION COMPARISON ===")
print("{:<25s} {:>10s} {:>10s} {:>10s}".format("Feature", "Real", "Synth", "Diff%"))
print("-" * 55)

keys = ['f0_mean', 'f0_std', 'f0_pct_voiced', 'zcr_mean', 'centroid_mean', 'duration']
for key in keys:
    r_vals = [f[key] for f in real_features if f[key] is not None]
    s_vals = [f[key] for f in synth_features if f[key] is not None]
    if not r_vals or not s_vals:
        continue
    r_mean = np.mean(r_vals)
    s_mean = np.mean(s_vals)
    diff_pct = abs(r_mean - s_mean) / max(abs(r_mean), 1e-6) * 100
    print("{:<25s} {:>10.3f} {:>10.3f} {:>9.1f}%".format(key, r_mean, s_mean, diff_pct))

# MFCC dimension-wise comparison
print("")
print("=== MFCC Dimension Comparison ===")
for dim in range(13):
    r_m = np.mean([f['mfcc_mean'][dim] for f in real_features])
    s_m = np.mean([f['mfcc_mean'][dim] for f in synth_features])
    print(f"MFCC{dim+1:2d}: real={r_m:7.2f} synth={s_m:7.2f}")

# Save summary
summary = {
    'n_synthetic': len(synth_features),
    'n_real': len(real_features),
}
with open(os.path.join(SYNTH_DIR, 'feature_comparison.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("")
print("Done: synthetic=" + str(len(synth_features)) + " real=" + str(len(real_features)))
