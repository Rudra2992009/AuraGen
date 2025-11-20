import torch
from typing import Dict
from model.neuron_web import ArtificialNeuronWeb
from utils.prompt_processor import PromptProcessor
from model.neural_architecture import AuraGenCore

class AuraFramework:
    """AuraGen full workflow: prompt -> neuron web -> neural chain -> output video"""
    def __init__(self, model_config: Dict):
        self.prompt_processor = PromptProcessor()
        self.neuron_web = ArtificialNeuronWeb()
        self.core = AuraGenCore(**model_config)
    def generate(self, text_prompt: str, image_refs: list = None, num_frames: int = 600) -> dict:
        prompt_data = self.prompt_processor.process_prompt(text_prompt, image_refs)
        prompt_tokens = prompt_data['tokens'].unsqueeze(0)
        image_neurons = prompt_data.get('image_refs', None)
        neuron_embedding = self.neuron_web(prompt_tokens, image_neurons)
        # Pass through core model, chaining neuron embedding as prompt
        outputs = self.core(neuron_embedding.unsqueeze(0), num_frames=num_frames)
        return outputs
