import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class ArtificialNeuronWeb(nn.Module):
    """Flexible graph-based neural web: connects chains & interlinks prompts."""
    def __init__(self, web_size: int = 1280, node_dim: int = 1024, depth: int = 4):
        super().__init__()
        self.web_size = web_size
        self.node_dim = node_dim
        self.depth = depth
        # Each layer is a set of graph nodes with attention links
        self.layers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=node_dim, nhead=8, dim_feedforward=node_dim*4
                ), num_layers=2
            ) for _ in range(depth)
        ])
        self.node_proj = nn.Linear(node_dim, node_dim)
    def forward(self, prompt_tokens: torch.Tensor, image_refs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        # Combine text & image prompts into web nodes
        batch = prompt_tokens.size(0)
        web_nodes = prompt_tokens.float().view(batch, -1, 1).repeat(1, 1, self.node_dim)
        if image_refs:
            # Simple aggregation of all reference images as additional nodes
            img_nodes = torch.cat([img.unsqueeze(0) for img in image_refs], dim=0).mean(dim=0, keepdim=True)
            web_nodes = torch.cat([web_nodes, img_nodes], dim=1)
        # Apply web interlinking through graph
        out = web_nodes
        for layer in self.layers:
            out = layer(out)
        out = self.node_proj(out.mean(dim=1))
        return out
