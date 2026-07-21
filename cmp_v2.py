import os, sys, glob
import numpy as np
import soundfile as sf
sys.path.insert(0, '/data/disk3/EMMA-BC/phaseB')
from multimodal_dataset import DAICWOZDataset

ds = DAICWOZDataset('/data/disk1/datasets/diac_woz', split='train', filter_interviewer=False, min_audio_sec=1.0, max_audio_sec=30)

def quick_feat(y, sr=16000):
    if len(y) < sr: return None
    if len(y) > sr * 30: y = y[:sr * 30]
    y = np.array(y).squeeze()
    mfcc = np.mean(np.abs(np.fft.rfft(y[:sr*10]))[:100])
    energy = np.sqrt(np.mean(y**2))
    zcr = np.mean(np.abs(np.diff(np.sign(y[:sr*5])))) / 2
    return {'mfcc': float(mfcc), 'energy': float(energy), 'zcr': float(zcr), 'dur': float(len(y)/sr)}

real_f = []
for i in range(len(ds)):
    item = ds[i]
    phq = item.get('phq_total', item.get('phq', 0))
    if phq < 10: continue
    y = item['audio']
    if hasattr(y, 'numpy'): y = y.numpy()
    y = np.array(y).squeeze()
    f = quick_feat(y)
    if f: real_f.append(f)
    if len(real_f) >= 20: break

synth_f = []
for fp in sorted(glob.glob('/data/disk3/EMMA-BC/synthetic/synth_*.wav')):
    y, sr = sf.read(fp)
    f = quick_feat(y, sr)
    if f: synth_f.append(f)

print("Real:", len(real_f), "Synth:", len(synth_f))
print("")
print("{:<15s} {:>10s} {:>10s} {:>10s}".format("Feature", "Real", "Synth", "Diff%"))
print("-" * 45)
for k in ['mfcc', 'energy', 'zcr', 'dur']:
    r = np.mean([x[k] for x in real_f])
    s = np.mean([x[k] for x in synth_f])
    pct = abs(r-s)/max(abs(r),1e-6)*100
    flag = "OK" if pct < 50 else "FIX"
    print("{:<15s} {:>10.4f} {:>10.4f} {:>9.1f}% [{}]".format(k, r, s, pct, flag))
