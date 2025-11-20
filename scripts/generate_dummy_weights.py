import json
import torch
from model.neural_architecture import create_auragen_model
from utils.model_weights import save_model_safetensors

if __name__ == "__main__":
    # Generate dummy weights (for testing the safetensors saving)
    config = {
        'dim': 2048,
        'num_chains': 12,
        'max_frames': 432000,
        'video_resolution': (1920, 1080)
    }
    model = create_auragen_model(config)
    save_model_safetensors(model, 'weights/aura-base.safetensors')
    print("Dummy weights saved.")
