import subprocess, os, json, random, numpy as np, soundfile as sf

OUT = "/data/disk3/EMMA-BC/synthetic"
os.makedirs(OUT, exist_ok=True)
EDGE = "/data/disk3/VorCAST/venv/bin/edge-tts"
ENV = dict(os.environ, HOME="/home/user")

TEXTS = [
    "I just don't feel like doing anything anymore.",
    "Some days I can barely get out of bed.",
    "I used to enjoy things but now everything feels exhausting.",
    "I feel like I'm letting everyone down.",
    "Sleep doesn't help. I wake up still tired and empty.",
    "I don't know why I feel this way but everything feels wrong.",
    "Sometimes I just sit and stare at the wall for hours.",
    "I've lost interest in everything I used to love.",
    "My thoughts keep racing and I can't make them stop.",
    "I feel like a burden to everyone around me.",
    "I try to be positive but the darkness always comes back.",
    "Eating feels like a chore. I forget to eat most days.",
    "I can't concentrate on anything. My mind is foggy.",
    "I feel so alone even when people are around me.",
    "Nothing brings me joy anymore. Not music, not anything.",
    "I'm tired all the time but can't sleep at night.",
    "I feel worthless. Like nothing I do matters.",
    "I push people away so they don't see how broken I am.",
    "Some days I feel okay for a few hours then it crashes down.",
    "I don't see the point in making plans for the future.",
    "I used to be so energetic and now I can barely function.",
    "I feel like I'm watching my life from outside my body.",
    "Every task feels overwhelming. Even small things.",
    "I've stopped answering calls. I just can't face people.",
    "I feel like crying all the time but the tears won't come.",
    "I don't remember what happiness feels like anymore.",
    "My body aches but the doctors say there's nothing wrong.",
    "I feel guilty all the time and I don't know why.",
    "I'm so irritable. The smallest things set me off.",
    "I've thought about just not waking up anymore.",
    "I feel stuck, like I'm trapped and can't get out.",
    "People tell me to cheer up but they don't understand.",
    "I fake being okay at work then collapse at home.",
    "I don't feel connected to anyone anymore.",
    "My memory is terrible lately. I forget everything.",
    "I feel like a failure. Everyone else has their life together.",
    "The mornings are the worst. Waking up feels like punishment.",
    "I'm afraid I'll feel this way forever.",
    "I don't talk about it because I don't want to worry people.",
    "Everything feels muted, like the world lost its color.",
    "I can't make decisions anymore. Even simple ones.",
    "I feel so heavy, like gravity is stronger for me.",
    "I used to dream about the future now I just survive today.",
    "No matter how much I sleep I'm always exhausted.",
    "I feel like I'm drowning and no one can see it.",
    "I don't recognize myself anymore. I've lost who I am.",
    "I get angry at myself for being like this.",
    "Sometimes I laugh but it's not real, just automatic.",
    "I want to get better but I don't have the energy to try.",
    "Every day feels exactly the same. Gray and empty.",
]

results = []
for i, text in enumerate(TEXTS):
    n = str(i).zfill(3)
    tf = "/tmp/tt_" + n + ".txt"
    wf = "/tmp/tw_" + n + ".wav"
    
    with open(tf, "w") as f:
        f.write(text)
    
    r = subprocess.run([EDGE, "--file", tf, "--voice", "en-US-AriaNeural",
                        "--rate", "-15%", "--write-media", wf],
                       capture_output=True, timeout=30, env=ENV)
    
    if r.returncode != 0 or not os.path.exists(wf):
        print("[" + n + "] FAILED rc=" + str(r.returncode))
        continue
    
    y, sr = sf.read(wf)
    
    # Depression modulation
    rms_val = np.sqrt(np.mean(y**2))
    if rms_val > 0:
        y = y / rms_val * 0.12
    
    for _ in range(random.randint(2, 3)):
        p = int(sr * random.uniform(0.3, 0.7))
        pos = random.randint(sr * 2, max(sr * 3, len(y) - sr))
        y = np.concatenate([y[:pos], np.zeros(p), y[pos:]])
    
    if len(y) > sr * 30:
        y = y[:sr * 30]
    
    out_path = os.path.join(OUT, "synth_" + n + ".wav")
    sf.write(out_path, y, sr)
    
    os.remove(tf)
    os.remove(wf)
    
    dur = len(y) / sr
    results.append({"path": out_path, "duration": round(dur, 1), "text": text[:80]})
    print("[" + n + "] dur=" + str(round(dur,1)) + "s")

with open(os.path.join(OUT, "metadata.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Done: " + str(len(results)) + " samples")
