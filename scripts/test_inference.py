import torch
import json
from model.neural_architecture import create_auragen_model
from utils.model_weights import load_model_safetensors

if __name__ == "__main__":
    # Test inference routine using dummy weights
    config = {
        'dim': 2048,
        'num_chains': 12,
        'max_frames': 432000,
        'video_resolution': (1920, 1080)
    }
    model = create_auragen_model(config)
    model = load_model_safetensors(create_auragen_model, config, 'weights/aura-base.safetensors')
    prompt = torch.randint(0, 50000, (1, 512))
    result = model(prompt, num_frames=600)
    print({ key: (v.shape if hasattr(v, 'shape') else type(v)) for key, v in result.items() })
