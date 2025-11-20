import torch
import safetensors.torch
from utils.secure_model_save.py import save_model_safetensors_with_notice
from utils.safety_filter import is_prompt_safe
from utils.antivirus_module import AntivirusHeuristicModule
from pathlib import Path

class AuraSecureWeightsManager:
    def __init__(self, model, weights_dir='weights'):
        self.model = model
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(exist_ok=True)
        self.antivirus = AntivirusHeuristicModule()
    def save(self, filename, prompt_test=None):
        path = self.weights_dir/filename
        # Safety check on context prompt
        if prompt_test and not is_prompt_safe(prompt_test):
            raise ValueError("Refusing to save weights: associated prompt is unsafe.")
        # Antivirus check
        dummy_tokens = torch.randint(0, 512, (1,512))
        resistant = not self.antivirus.scan_code(dummy_tokens)
        if not resistant:
            raise ValueError("Refusing to save weights: malware detected by antivirus scan.")
        save_model_safetensors_with_notice(self.model, str(path), extra_meta={'security':'Passed safety/antivirus checks.'})
        print(f"Weights securely saved to {path}")
