import torch
import safetensors.torch
from pathlib import Path

# Functions for managing binary model checkpoints

def save_binary_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")

def load_binary_checkpoint(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    print(f"Checkpoint loaded from {path}")
    return model

# Safetensors management

def save_safetensors_checkpoint(model, path):
    safetensors.torch.save_file(model.state_dict(), path)
    print(f"Safetensors checkpoint saved: {path}")

def load_safetensors_checkpoint(model_class, config, path):
    state_dict = safetensors.torch.load_file(path)
    model = model_class(**config)
    model.load_state_dict(state_dict)
    print(f"Safetensors checkpoint loaded: {path}")
    return model
