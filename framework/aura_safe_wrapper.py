import torch
from utils.safety_filter import is_prompt_safe

class AuraSafeFramework:
    """Safe pipeline wrapper: rejects any prompt containing illegal/deepfake/explicit requests."""
    def __init__(self, aura_framework):
        self.aura_framework = aura_framework
    def generate(self, text_prompt: str, image_refs: list = None, num_frames: int = 600):
        # Safety check on prompt
        if not is_prompt_safe(text_prompt):
            raise ValueError("Prompt contains prohibited content (deepfake/sex/explicit). Generation declined.")
        return self.aura_framework.generate(text_prompt, image_refs, num_frames)
