#!/usr/bin/env python3
"""Generate synthetic depressed speech - synchronous version."""
import os, json, random, subprocess, time
import numpy as np
import soundfile as sf

OUT_DIR = "/data/disk3/EMMA-BC/synthetic"
os.makedirs(OUT_DIR, exist_ok=True)

TEXTS = [
    "I just don't feel like doing anything anymore. Everything seems pointless.",
    "Some days I can barely get out of bed. It's like my body is made of lead.",
    "I used to enjoy cooking but now even making a sandwich feels exhausting.",
    "I feel like I'm letting everyone down. My family deserves better.",
    "Sleep doesn't help. I wake up and I'm still tired, still empty.",
    "I don't know why I feel this way. There's nothing wrong but everything feels wrong.",
    "Sometimes I just sit and stare at the wall for hours.",
    "I've lost interest in everything. Even the things I used to love.",
    "My thoughts keep racing and I can't make them stop. It's exhausting.",
    "I feel like a burden to everyone around me. They'd be better off without me.",
    "I try to be positive but it doesn't work. The darkness always comes back.",
    "Eating feels like a chore. I've lost weight because I just forget to eat.",
    "I can't concentrate on anything. My mind is foggy all the time.",
    "I feel so alone even when there are people around me.",
    "Nothing brings me joy anymore. Not music, not movies, not people.",
    "I'm tired all the time but when I try to sleep, my mind won't shut up.",
    "I feel worthless. Like nothing I do matters.",
    "I push people away because I don't want them to see how broken I am.",
    "Some days I feel okay for a few hours and then it all comes crashing down.",
    "I don't see the point in making plans. The future feels empty.",
    "I used to be so energetic and now I can barely function.",
    "I feel like I'm watching my life from outside my body.",
    "Every task feels overwhelming. Even brushing my teeth is hard.",
    "I've stopped answering calls and messages. I just can't face people.",
    "I feel like crying all the time but the tears won't come.",
    "I don't remember what happiness feels like anymore.",
    "My body aches all over but the doctors say there's nothing wrong.",
    "I feel guilty all the time and I don't even know why.",
    "I'm so irritable. The smallest things set me off.",
    "I've thought about ending it. Not planning it, just wishing I wouldn't wake up.",
    "I feel stuck, like I'm trapped in quicksand and can't get out.",
    "People tell me to cheer up but they don't understand.",
    "I fake being okay at work but when I get home I just collapse.",
    "I don't feel connected to anyone anymore. It's like there's a wall.",
    "My memory is terrible lately. I forget everything.",
    "I feel like a failure. Everyone else seems to have their life together.",
    "The mornings are the worst. Waking up feels like a punishment.",
    "I'm afraid I'll feel this way forever.",
    "I don't talk about it because I don't want to worry people.",
    "Everything feels muted, like the world has lost its color.",
    "I can't make decisions anymore. Even choosing what to eat is too much.",
    "I feel so heavy, like gravity is stronger for me than everyone else.",
    "I used to dream about the future and now I just try to get through today.",
    "No matter how much I sleep, I'm always exhausted.",
    "I feel like I'm drowning and no one can see it.",
    "I've lost my sense of who I am. I don't recognize myself anymore.",
    "I get angry at myself for being like this. It feels so weak.",
    "Sometimes I laugh but it's not real. It's just automatic.",
    "I want to get better but I don't have the energy to try.",
    "Every day feels exactly the same. Gray and empty.",
]

EDGE_TTS = "/data/disk3/VorCAST/venv/bin/edge-tts"
results = []

for i, text in enumerate(TEXTS):
    tmp = f"/tmp/synth_{i}.wav"
    out_path = os.path.join(OUT_DIR, f"synth_{i:03d}.wav")
    
    # Generate TTS
    r = subprocess.run([EDGE_TTS, "--text", text, "--voice", "en-US-AriaNeural",
                        "--rate", "-15%", "--write-media", tmp],
                       capture_output=True, timeout=30)
    if not os.path.exists(tmp):
        print(f"[{i+1:02d}/50] FAILED TTS")
        continue
    
    # Load audio
    y, sr = sf.read(tmp)
    if sr != 16000:
        import librosa; y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
    
    # Depression modulation
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        y = y / rms * 0.12
    
    # Insert pauses
    for _ in range(random.randint(2, 3)):
        pause = int(sr * random.uniform(0.3, 0.7))
        pos = random.randint(sr * 2, max(sr * 3, len(y) - sr))
        y = np.concatenate([y[:pos], np.zeros(pause), y[pos:]])
    
    # Max length
    if len(y) > sr * 30:
        y = y[:sr * 30]
    
    sf.write(out_path, y, sr)
    os.remove(tmp)
    
    dur = len(y) / sr
    results.append({"path": out_path, "duration": round(dur, 1), "text": text[:80]})
    print(f"[{i+1:02d}/50] dur={dur:.1f}s | {text[:60]}...")

with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"
Done: {len(results)}/{len(TEXTS)} samples saved to {OUT_DIR}/")
