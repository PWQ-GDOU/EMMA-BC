
import torch
from funasr import AutoModel
print("Loading emotion2vec+ large...")
model = AutoModel(model="iic/emotion2vec_plus_large", trust_remote_code=True, device="cuda:0")
print(f"Loaded: {type(model)}")
waveform = torch.randn(1, 16000)
with torch.no_grad():
    result = model(waveform, output_dir=None)
print(f"Result: {list(result.keys()) if isinstance(result, dict) else type(result)}")
# Check feature dim
if hasattr(model.model, 'encoder_embed_dim'):
    print(f"encoder_embed_dim={model.model.encoder_embed_dim}")
print("DONE")
