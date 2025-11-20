import os
from pathlib import Path
import torch
from model.neural_architecture import create_auragen_model
from utils.model_weights import save_model_safetensors, load_model_safetensors

def export_model_pt(config, out_path):
    model = create_auragen_model(config)
    scripted = torch.jit.script(model)
    scripted.save(out_path)
    print(f"Scripted model saved: {out_path}")

if __name__ == "__main__":
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    config = {
        'dim': 2048,
        'num_chains': 12,
        'max_frames': 432000,
        'video_resolution': (1920, 1080)
    }
    out_path = str(weights_dir / "auragen_scripted.pt")
    export_model_pt(config, out_path)
