import os, sys, json, glob
import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, '/data/disk3/EMMA-BC/phaseB')
from multimodal_dataset import DAICWOZDataset

SYNTH_DIR = '/data/disk3/EMMA-BC/synthetic'

ds = DAICWOZDataset('/data/disk1/datasets/diac_woz', split='train', filter_interviewer=False, min_audio_sec=1.0, max_audio_sec=30)

def feats(y, sr):
    if len(y) < sr: return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    f0, vf, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    f0v = f0[vf] if np.any(vf) else np.array([100.0])
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return {
        'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
        'f0_mean': float(np.mean(f0v)),
        'f0_std': float(np.std(f0v)),
        'f0_voiced': float(np.mean(vf)),
        'zcr_mean': float(np.mean(zcr)),
        'dur': float(len(y)/sr),
    }

real_f = []
n_dep = 0
for i in range(len(ds)):
    item = ds[i]
    phq = item.get('phq_total', item.get('phq', 0))
    if phq < 10:
        continue
    n_dep += 1
    if n_dep > 20:
        break
    
    y = item['audio']
    sr = 16000
    if len(y) > sr * 30:
        y = y[:sr * 30]
    y_np = y.numpy() if hasattr(y, 'numpy') else y
    if len(y_np.shape) > 1:
        y_np = y_np.squeeze()
    f = feats(y_np, sr)
    if f:
        real_f.append(f)

print("Real depressed:", len(real_f))

synth_f = []
for fpath in sorted(glob.glob(SYNTH_DIR + '/synth_*.wav')):
    y, sr = sf.read(fpath)
    f = feats(y, sr)
    if f:
        synth_f.append(f)
print("Synth:", len(synth_f))

if real_f and synth_f:
    print("")
    print("{:<15s} {:>10s} {:>10s}".format("Feature", "Real", "Synth"))
    print("-" * 35)
    for k in ['f0_mean', 'f0_std', 'f0_voiced', 'zcr_mean', 'dur']:
        r = np.mean([x[k] for x in real_f])
        s = np.mean([x[k] for x in synth_f])
        print("{:<15s} {:>10.3f} {:>10.3f}".format(k, r, s))
    print("")
    print("MFCC first 5 dims:")
    for d in range(5):
        r = np.mean([x['mfcc_mean'][d] for x in real_f])
        s = np.mean([x['mfcc_mean'][d] for x in synth_f])
        print("  MFCC" + str(d+1) + ": real=" + str(round(r,2)) + " synth=" + str(round(s,2)))
