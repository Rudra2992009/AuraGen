import torch
import torch.nn as nn
from typing import List, Dict

class StyleTransferEngine(nn.Module):
    """Dynamically blend art styles, genres, and film looks."""
    def __init__(self, style_dim: int = 512, base_styles: int = 5):
        super().__init__()
        self.style_dim = style_dim
        self.base_styles = nn.Embedding(base_styles, style_dim)
        self.mixer = nn.Sequential(
            nn.Linear(style_dim * 2, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim)
        )
    def forward(self, frame: torch.Tensor, style_ids: List[int]) -> torch.Tensor:
        batch, c, h, w = frame.shape
        mixed_style = self.base_styles(torch.tensor(style_ids).to(frame.device)).mean(dim=0)
        style_tensor = mixed_style.view(1, self.style_dim, 1, 1).repeat(batch, 1, h, w)
        mixed = self.mixer(torch.cat([frame, style_tensor], dim=1).view(batch, -1, h*w))
        return mixed.view(batch, self.style_dim, h, w)
