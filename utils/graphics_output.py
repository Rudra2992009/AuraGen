import torch
from PIL import Image
import numpy as np
from pathlib import Path

def save_graphics(graphics_latent, filename, res=(352,288)):
    """Save individual graphic frame as PNG from tensor."""
    frame = graphics_latent.squeeze().detach().cpu().numpy().reshape(res[0], res[1], 3)
    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(frame)
    img.save(filename)
    print(f"Graphic frame saved: {filename}")

if __name__ == "__main__":
    graphics_latent = torch.randn(res[0]*res[1]*3)
    save_graphics(graphics_latent, 'output/dummy_graphic.png')
