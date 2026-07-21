#!/bin/bash
export HOME=/home/user
EDGE=/data/disk3/VorCAST/venv/bin/edge-tts
OUT=/data/disk3/EMMA-BC/synthetic
mkdir -p $OUT

gen() {
  local num=$1 text=$2
  local n=$(printf "%03d" $num)
  echo "$text" > "/tmp/st_$n.txt"
  $EDGE --file "/tmp/st_$n.txt" --voice en-US-AriaNeural --rate "-15%" --write-media "/tmp/sw_$n.wav" 2>/dev/null
  if [ -f "/tmp/sw_$n.wav" ]; then
    python3 -c "
import soundfile as sf, numpy as np, random
y, sr = sf.read('/tmp/sw_$n.wav')
rms = np.sqrt(np.mean(y**2))
if rms > 0: y = y / rms * 0.12
for _ in range(random.randint(2,3)):
    p = int(sr * random.uniform(0.3, 0.7))
    pos = random.randint(sr*2, max(sr*3, len(y)-sr))
    y = np.concatenate([y[:pos], np.zeros(p), y[pos:]])
if len(y) > sr*30: y = y[:sr*30]
sf.write('$OUT/synth_$n.wav', y, sr)
print(f'[$n] dur={len(y)/sr:.1f}s')
" 2>/dev/null
    rm -f "/tmp/sw_$n.wav"
  else
    echo "[$n] FAILED"
  fi
  rm -f "/tmp/st_$n.txt"
}

gen 0 "I just don't feel like doing anything anymore."
gen 1 "Some days I can barely get out of bed."
gen 2 "I used to enjoy things but now everything feels exhausting."
gen 3 "I feel like I'm letting everyone down."
gen 4 "Sleep doesn't help. I wake up still tired and empty."
gen 5 "I don't know why I feel this way but everything feels wrong."
gen 6 "Sometimes I just sit and stare at the wall for hours."
gen 7 "I've lost interest in everything I used to love."
gen 8 "My thoughts keep racing and I can't make them stop."
gen 9 "I feel like a burden to everyone around me."
echo "=== Batch 1 done ==="
