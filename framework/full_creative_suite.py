import torch
from typing import Dict
from framework.hyperrealistic_pipeline import AuraHyperRealisticPipeline
from framework.creative_enhancer import CreativeEnhancer

class AuraFullCreativeSuite:
    """Full pipeline: hyper-realistic video + creative style & music adaptation."""
    def __init__(self, config: dict):
        self.realistic_pipeline = AuraHyperRealisticPipeline(config)
        self.creative_enhancer = CreativeEnhancer()
    def generate(self, prompt: torch.Tensor, style_ids: list, num_frames: int = 432000) -> Dict:
        pipeline_outputs = self.realistic_pipeline.generate(prompt, num_frames=num_frames)
        enhanced = self.creative_enhancer.enhance(
            pipeline_outputs['video'],
            style_ids,
            prompt,
            scene_boundaries=None
        )
        return {
            **pipeline_outputs,
            **enhanced
        }
