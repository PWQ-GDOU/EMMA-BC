#!/usr/bin/env python3
"""Extract WavLM+BERT+OpenFace features for MODMA and DAIC-WOZ."""
import os, sys, json, numpy as np, torch
from pathlib import Path
import soundfile as sf, pandas as pd
from tqdm import tqdm

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
DEVICE = torch.device('cuda:0')
SR, MAX_SEC = 16000, 30

def load_audio(path, max_sec=MAX_SEC):
    audio, _ = sf.read(str(path))
    audio = torch.FloatTensor(audio)
    max_s = int(max_sec * SR)
    if len(audio) > max_s: audio = audio[:max_s]
    else: audio = torch.cat([audio, torch.zeros(max_s - len(audio))])
    return audio

def extract_daic():
    """Extract features for DAIC-WOZ train+dev participants."""
    from transformers import WavLMModel, BertModel, BertTokenizer
    print('Loading models...')
    wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base-plus').to(DEVICE).eval()
    bert = BertModel.from_pretrained('bert-base-uncased').to(DEVICE).eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    data_dir = Path('/data/disk1/datasets/diac_woz')
    all_data = []
    for split in ['train', 'dev']:
        df = pd.read_csv(data_dir / 'labels' / f'{split}_split.csv')
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'DAIC {split}'):
            pid = row['Participant_ID']
            ap = data_dir / f'{pid}_P' / f'{pid}_AUDIO.wav'
            if not ap.exists(): continue
            # Audio
            audio = load_audio(ap).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                af = wavlm(audio).last_hidden_state.mean(dim=1).cpu().squeeze(0)
            # Text
            tp = data_dir / f'{pid}_P' / f'{pid}_Transcript.csv'
            text = ''
            if tp.exists():
                try:
                    tdf = pd.read_csv(tp)
                    text = ' '.join(tdf['Text'].dropna().tolist())[:2000]
                except: pass
            if text:
                tok = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(DEVICE)
                with torch.no_grad():
                    tf = bert(**tok).last_hidden_state.mean(dim=1).cpu().squeeze(0)
            else:
                tf = torch.zeros(768)
            # Video
            vp = data_dir / f'{pid}_P' / 'features' / f'{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv'
            vf = torch.zeros(49)
            if vp.exists():
                try:
                    vdf = pd.read_csv(vp)
                    cols = [c for c in vdf.columns if c not in ['frame','timestamp','confidence','success']]
                    vf = torch.FloatTensor(vdf[cols].values).mean(dim=0)
                except: pass
            # Label
            label = float(row.get('PHQ_Score', 0))
            all_data.append({'pid': pid, 'audio': af.numpy(), 'text': tf.numpy(), 'video': vf.numpy(), 'label': label, 'split': split})
    
    np.savez('results/daic_features.npz', data=np.array(all_data, dtype=object))
    print(f'DAIC: {len(all_data)} participants saved')

def extract_modma():
    """Extract WavLM audio features for MODMA."""
    from transformers import WavLMModel
    print('Loading WavLM...')
    wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base-plus').to(DEVICE).eval()
    
    data_dir = Path('/data/disk1/datasets/modma/extracted/audio_lanzhou_2015')
    xlsx = data_dir / 'subjects_information_audio_lanzhou_2015.xlsx'
    df = pd.read_excel(xlsx)
    df.columns = ['sid','type','age','gender','edu','PHQ9','CTQ','LES','SSRS','GAD7','PSQI','_a','_b']
    df['sid'] = df['sid'].astype(str).str.zfill(8)
    
    all_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='MODMA'):
        sid = row['sid']
        subj = data_dir / sid
        if not subj.exists(): continue
        wavs = sorted(subj.glob('*.wav'))
        segs = []
        for w in wavs:
            try: a, _ = sf.read(str(w)); segs.append(torch.FloatTensor(a))
            except: pass
        if not segs: continue
        audio = torch.cat(segs)[:int(MAX_SEC * SR)]
        if len(audio) < 16000: audio = torch.cat([audio, torch.zeros(16000 - len(audio))])
        audio = audio.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = wavlm(audio).last_hidden_state.mean(dim=1).cpu().squeeze(0)
        label = float(row['PHQ9'])
        all_data.append({'sid': sid, 'audio': feat.numpy(), 'label': label, 'type': row['type']})
    
    np.savez('results/modma_features.npz', data=np.array(all_data, dtype=object))
    print(f'MODMA: {len(all_data)} subjects saved')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='all', choices=['daic','modma','all'])
    args = p.parse_args()
    if args.dataset in ('daic','all'): extract_daic()
    if args.dataset in ('modma','all'): extract_modma()
