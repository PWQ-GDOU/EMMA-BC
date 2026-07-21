import subprocess, os, json, random
import numpy as np
import soundfile as sf
import librosa

BASE = '/data/disk3/EMMA-BC'
TEXTS = json.load(open(BASE + '/synth_texts.json'))
OUT = BASE + '/synthetic'
os.makedirs(OUT, exist_ok=True)
EDGE = '/data/disk3/VorCAST/venv/bin/edge-tts'
ENV = {'HOME': '/home/user', 'PATH': '/usr/bin:/usr/local/bin'}
results = []

for i, text in enumerate(TEXTS):
    n = str(i).zfill(3)
    tf = '/tmp/st_' + n + '.txt'
    wf = '/tmp/sw_' + n + '.wav'
    
    with open(tf, 'w') as f:
        f.write(text)
    r = subprocess.run([EDGE, '--file', tf, '--voice', 'en-US-AriaNeural',
                        '--write-media', wf],
                       capture_output=True, timeout=30, env=ENV)
    
    if r.returncode != 0 or not os.path.exists(wf):
        print('[' + n + '] FAILED rc=' + str(r.returncode))
        continue
    
    y, sr = sf.read(wf)
    
    # Slow down speech by 15% (depression-like)
    y = librosa.effects.time_stretch(y=y, rate=0.85)
    
    # Lower volume
    rms_val = np.sqrt(np.mean(y**2))
    if rms_val > 0:
        y = y / rms_val * 0.12
    
    # Insert random pauses
    for _ in range(random.randint(2, 3)):
        p = int(sr * random.uniform(0.3, 0.7))
        pos = random.randint(sr * 2, max(sr * 3, len(y) - sr))
        y = np.concatenate([y[:pos], np.zeros(p), y[pos:]])
    
    if len(y) > sr * 30:
        y = y[:sr * 30]
    
    op = OUT + '/synth_' + n + '.wav'
    sf.write(op, y, sr)
    dur = len(y) / sr
    results.append({'path': op, 'duration': round(dur, 1), 'text': text[:80]})
    print('[' + n + '] dur=' + str(round(dur, 1)) + 's')
    os.remove(tf)
    os.remove(wf)

with open(OUT + '/metadata.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Done: ' + str(len(results)) + ' samples')
