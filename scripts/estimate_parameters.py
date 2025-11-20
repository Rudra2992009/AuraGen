# Parameter Estimation Utility for AuraGen
import torch
from model.neural_architecture import AuraGenCore

MODEL_CONFIGS = {
    'AuraGen-Small':   {'dim': 768,  'num_chains': 6,  'max_frames': 432000, 'video_resolution': (512, 320)},
    'AuraGen-Base':    {'dim': 2048, 'num_chains': 12, 'max_frames': 432000, 'video_resolution': (1920, 1080)},
    'AuraGen-Large':   {'dim': 4096, 'num_chains': 32, 'max_frames': 432000, 'video_resolution': (3840, 2160)},
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    for name, cfg in MODEL_CONFIGS.items():
        model = AuraGenCore(**cfg)
        n_params = count_parameters(model)
        print(f"{name}: {n_params/1e9:.2f}B parameters ({n_params:,})")
