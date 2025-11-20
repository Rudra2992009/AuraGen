import torch
import safetensors.torch

def save_model_safetensors(model, path):
    state_dict = model.state_dict()
    safetensors.torch.save_file(state_dict, path)


def load_model_safetensors(model_class, config, path):
    state_dict = safetensors.torch.load_file(path)
    model = model_class(**config)
    model.load_state_dict(state_dict)
    return model

# Example usage (in training script):
# save_model_safetensors(model, "weights/aura-base.safetensors")
# model = load_model_safetensors(AuraGenCore, config, "weights/aura-base.safetensors")
