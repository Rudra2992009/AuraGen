import torch
import numpy as np
from pathlib import Path
import json

class ModelRegistry:
    """Registry for managing multiple model variants and checkpoints."""
    
    def __init__(self, registry_path: str = 'weights/registry.json'):
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> dict:
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        self.registry_path.parent.mkdir(exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, name: str, path: str, config: dict, metadata: dict = None):
        """Register a model checkpoint."""
        self.registry[name] = {
            'path': path,
            'config': config,
            'metadata': metadata or {}
        }
        self._save_registry()
        print(f"Model '{name}' registered.")
    
    def get_model_info(self, name: str) -> dict:
        """Get model information."""
        return self.registry.get(name, None)
    
    def list_models(self) -> list:
        """List all registered models."""
        return list(self.registry.keys())

if __name__ == "__main__":
    registry = ModelRegistry()
    registry.register_model(
        'aura-base',
        'weights/aura-base.safetensors',
        {'dim': 2048, 'num_chains': 12},
        {'version': '1.0', 'trained_steps': 0}
    )
    print("Registered models:", registry.list_models())
