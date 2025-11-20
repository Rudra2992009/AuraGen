import torch
from model.neural_architecture import create_auragen_model
from utils.secure_weights_manager import AuraSecureWeightsManager

if __name__ == "__main__":
    cfg = {
      'dim': 2048,
      'num_chains': 12,
      'max_frames': 432000,
      'video_resolution': (1920,1080)
    }
    model = create_auragen_model(cfg)
    manager = AuraSecureWeightsManager(model)
    manager.save('aura-secure-base.safetensors', prompt_test="A student makes a scientific video.")
