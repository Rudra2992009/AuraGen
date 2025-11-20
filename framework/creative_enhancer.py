import torch
from typing import Dict
from model.style_transfer import StyleTransferEngine
from model.adaptive_soundtrack import AdaptiveSoundtrackModel

class CreativeEnhancer:
    """Add extra creative flexibility: film styles & adaptive soundtracks."""
    def __init__(self):
        self.style_engine = StyleTransferEngine()
        self.soundtrack_model = AdaptiveSoundtrackModel()
    def enhance(self, video: torch.Tensor, style_ids, context: torch.Tensor, scene_boundaries=None) -> Dict:
        styled_video = self.style_engine(video, style_ids)
        adapted_music = self.soundtrack_model(context, scene_boundaries)
        return {
            'styled_video': styled_video,
            'music': adapted_music
        }
