import os, json, glob
import soundfile as sf
import numpy as np

SYNTH_DIR = '/data/disk3/EMMA-BC/synthetic'
# Try using UTMOS if available
try:
    from utmos import UTMOS
    model = UTMOS()
    has_utmos = True
    print("UTMOS model loaded")
except ImportError:
    has_utmos = False
    print("UTMOS not available, using heuristic metrics")

def estimate_snr(y):
    """Estimate SNR based on speech/silence segmentation."""
    rms = np.sqrt(np.mean(y**2))
    if rms < 1e-6:
        return 0
    # Simple threshold-based: top 10% is signal, bottom 50% is noise
    sorted_vals = np.sort(np.abs(y))
    noise = sorted_vals[:len(y)//2]
    noise_rms = np.sqrt(np.mean(noise**2)) if len(noise) > 0 else 1e-6
    return 20 * np.log10(rms / noise_rms)

def speech_ratio(y, sr):
    """Ratio of frames with energy above threshold."""
    frame_len = int(sr * 0.025)
    energy = np.array([np.mean(y[i:i+frame_len]**2) for i in range(0, len(y)-frame_len, frame_len)])
    if len(energy) < 2:
        return 0
    threshold = np.median(energy) * 0.3
    return np.mean(energy > threshold)

results = []
files = sorted(glob.glob(os.path.join(SYNTH_DIR, 'synth_*.wav')))

for f in files:
    y, sr = sf.read(f)
    name = os.path.basename(f)
    
    # Compute metrics
    snr = estimate_snr(y)
    sr_ratio = speech_ratio(y, sr)
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y**2)))
    
    # UTMOS if available
    utmos_score = None
    if has_utmos:
        try:
            utmos_score = float(model.predict(y, sr))
        except:
            pass
    
    status = 'PASS' if duration > 2 and sr_ratio > 0.3 and snr > 5 else 'FAIL'
    
    results.append({
        'file': name,
        'duration': round(duration, 1),
        'snr_db': round(snr, 1),
        'speech_ratio': round(sr_ratio, 2),
        'utmos': round(utmos_score, 2) if utmos_score else None,
        'rms': round(rms, 4),
        'status': status,
    })
    print(name + ' dur=' + str(round(duration,1)) + 's snr=' + str(round(snr,1)) + 'dB sr=' + str(round(sr_ratio,2)) + ' utmos=' + (str(round(utmos_score,2)) if utmos_score else 'N/A') + ' [' + status + ']')

with open(os.path.join(SYNTH_DIR, 'quality.json'), 'w') as f:
    json.dump(results, f, indent=2)

pass_count = sum(1 for r in results if r['status'] == 'PASS')
print('')
print('Passed: ' + str(pass_count) + '/' + str(len(results)))
print('Avg duration: ' + str(round(np.mean([r['duration'] for r in results]), 1)) + 's')
print('Avg SNR: ' + str(round(np.mean([r['snr_db'] for r in results]), 1)) + 'dB')
