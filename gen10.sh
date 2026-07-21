#!/bin/bash
export HOME=/home/user
EDGE=/data/disk3/VorCAST/venv/bin/edge-tts
OUT=/data/disk3/EMMA-BC/synthetic
mkdir -p $OUT

for i in $(seq 0 9); do
    NUM=$(printf "%03d" $i)
    $EDGE --file /tmp/st_$NUM.txt --voice en-US-AriaNeural --rate "-15%" --write-media /tmp/sw_$NUM.wav 2>/dev/null
    if [ -f /tmp/sw_$NUM.wav ]; then
        python3 -c "
import soundfile as sf, numpy as np, random
y, sr = sf.read('/tmp/sw_$NUM.wav')
rms = np.sqrt(np.mean(y**2))
if rms > 0: y = y / rms * 0.12
for _ in range(random.randint(2,3)):
    p = int(sr * random.uniform(0.3, 0.7))
    pos = random.randint(sr*2, max(sr*3, len(y)-sr))
    y = np.concatenate([y[:pos], np.zeros(p), y[pos:]])
if len(y) > sr*30: y = y[:sr*30]
sf.write('$OUT/synth_$NUM.wav', y, sr)
print(f'[$NUM] dur={len(y)/sr:.1f}s')
"
    else
        echo "[$NUM] FAILED"
    fi
    rm -f /tmp/st_$NUM.txt /tmp/sw_$NUM.wav
done
echo Done
