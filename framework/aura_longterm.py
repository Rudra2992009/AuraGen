import torch
from model.longterm_imagination import LongTermImagination
from utils.prompt_processor import PromptProcessor
from model.neural_architecture import AuraGenCore

class AuraLongTerm:
    """Full pipeline with expanded imagination for hyper-real long video."""
    def __init__(self, model_config):
        self.prompt_processor = PromptProcessor()
        self.imagination = LongTermImagination()
        self.core = AuraGenCore(**model_config)
    def generate(self, text_prompt: str, image_refs: list = None, duration: int = 7200):
        prompt_data = self.prompt_processor.process_prompt(text_prompt, image_refs)
        prompt_tokens = prompt_data['tokens'].unsqueeze(0)
        scene_emb = self.imagination(prompt_tokens, num_frames=duration*24)
        outputs = self.core(scene_emb, num_frames=duration*24)
        return outputs
