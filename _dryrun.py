import sys
sys.path.insert(0, 'phaseB')
from multimodal_dataset import DAICWOZDataset, MODMADataset, CombinedMultimodalDataset, collate_combined
from torch.utils.data import DataLoader

print("Loading DAIC-WOZ...")
daic = DAICWOZDataset('/data/disk1/datasets/diac_woz', split='train')
print("Loading MODMA...")
modma = MODMADataset('/data/disk1/datasets/modma/extracted', '/data/disk1/datasets/modma/extracted/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx')
print("Combining...")
combined = CombinedMultimodalDataset(daic, modma)
train_ds, val_ds = combined.split_by_subject(val_ratio=0.15, seed=42)

print()
print("Sample check:")
for i in [0, len(daic)]:
    sample = combined[i]
    print(f"  [{i}] source={sample['source']} audio={sample['audio'].shape} phq={sample['phq']:.1f} has_text={sample['has_text']}")

print()
print("Batch test:")
loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_combined)
batch = next(iter(loader))
print(f"  audio: {batch['audio'].shape}")
print(f"  texts: {len(batch['texts'])} items")
print(f"  phq: {batch['phq_total'].tolist()}")
print(f"  has_text: {batch['has_text'].tolist()}")
print()
print("DRY-RUN OK")
